# Ray Task Submission 完整交互逻辑

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [Normal Task 提交流程](#2-normal-task-提交流程)
3. [Actor Task 提交流程](#3-actor-task-提交流程)
4. [Dependency Resolution 机制](#4-dependency-resolution-机制)
5. [Task Manager 生命周期管理](#5-task-manager-生命周期管理)
6. [Raylet 侧调度与执行](#6-raylet-侧调度与执行)
7. [Task Receiver 接收与执行](#7-task-receiver-接收与执行)
8. [关键数据结构](#8-关键数据结构)
9. [错误处理与重试机制](#9-错误处理与重试机制)
10. [gRPC 接口定义](#10-grpc-接口定义)

---

## 1. 整体架构概览

Ray 的 Task Submission 系统由以下核心组件协作完成：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Language Frontend                            │
│                    (Python / Java / C++)                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ 调用 remote function / actor method
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CoreWorker                                  │
│  ┌────────────────┐  ┌───────────────────┐  ┌──────────────────┐  │
│  │ TaskManager    │  │ DependencyResolver│  │  TaskSubmitter   │  │
│  │ (生命周期追踪) │  │ (依赖解析)        │  │                  │  │
│  └────────────────┘  └───────────────────┘  │  ┌────────────┐ │  │
│                                              │  │NormalTask  │ │  │
│                                              │  │Submitter   │ │  │
│                                              │  └────────────┘ │  │
│                                              │  ┌────────────┐ │  │
│                                              │  │ActorTask   │ │  │
│                                              │  │Submitter   │ │  │
│                                              │  └────────────┘ │  │
│                                              └──────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ gRPC: RequestWorkerLease / PushTask
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Raylet (NodeManager)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │ClusterLeaseMgr   │  │LocalLeaseMgr     │  │ WorkerPool       │ │
│  │(分布式调度决策)  │  │(本地资源分配)    │  │(Worker 生命周期) │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            ClusterResourceScheduler                          │  │
│  │            (集群资源视图 + 调度算法)                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Grant Lease -> Worker Address
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Remote Worker CoreWorker                          │
│  ┌──────────────────┐                                               │
│  │ TaskReceiver     │ → SchedulingQueue → TaskHandler (语言回调)   │
│  │ (接收并执行)     │                                               │
│  └──────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 两种 Task 类型的关键差异

| 特性 | Normal Task | Actor Task |
|------|------------|------------|
| 执行目标 | 临时租用的 Worker | 固定的 Actor Worker |
| 资源获取 | 通过 Raylet Worker Lease | Actor 创建时已分配 |
| 任务顺序 | 无顺序保证 | 有顺序号保证（可配置顺序/乱序） |
| 队列组织 | 按 SchedulingKey（资源形态+函数描述） | 按 ActorID |
| 连接管理 | 短连接，任务完成归还 Worker | 长连接，Actor 生命周期管理 |
| 容错 | Worker 死亡后重新调度 | Actor 重启后重连，需处理状态恢复 |

---

## 2. Normal Task 提交流程

### 2.1 完整调用链

```
CoreWorker::SubmitTask()
  │
  ├─ 1. 构建 TaskSpecification
  │
  ├─ 2. TaskManager::AddPendingTask()
  │     └─ 注册 pending task，创建 return ObjectRef
  │     └─ 返回 returned_refs 给前端
  │
  ├─ 3. NormalTaskSubmitter::SubmitTask(task_spec)
  │     │
  │     ├─ 3a. LocalDependencyResolver::ResolveDependencies()
  │     │       │ 异步解析依赖
  │     │       ├─ 从 in_memory_store_ 获取 local dependencies
  │     │       ├─ 标记 remote dependencies (Plasma ObjectRefs)
  │     │       └─ 回调: ResolveDependenciesCallback()
  │     │
  │     ├─ 3b. [依赖解析完成后]
  │     │       │
  │     │       ├─ 计算SchedulingKey (resource_shape + function_descriptor)
  │     │       ├─ 将 task 加入 scheduling_key_entries_[key].task_queue
  │     │       ├─ ReportWorkerBacklog() 向 Raylet 报告队列深度
  │     │       │
  │     │       └─ RequestNewWorkerIfNeeded()
  │     │           │
  │     │           ├─ 检查 LeaseRequestRateLimiter 是否允许新请求
  │     │           ├─ RayletClient::RequestWorkerLease()
  │     │           │   gRPC → NodeManagerService::RequestWorkerLease
  │     │           └─ 记录 pending_lease_requests[lease_id]
  │     │
  │     ├─ 3c. [Worker Lease 授予后]
  │     │       │
  │     │       └─ OnWorkerIdle(worker_address, task_spec, ...)
  │     │           │
  │     │           ├─ 从 task_queue 取出下一个 task
  │     │           ├─ 标记 worker is_busy = true
  │     │           │
  │     │           └─ PushNormalTask()
  │     │               │
  │     │               ├─ CoreWorkerClient::PushNormalTask()
  │     │               │   gRPC → CoreWorkerService::PushTask
  │     │               └─ 注册 inflight callback
  │     │
  │     ├─ 3d. [PushTask Reply 回来后]
  │     │       │
  │     │       └─ HandlePushTaskReply()
  │     │           │
  │     │           ├─ TaskManager::CompletePendingTask()
  │     │           ├─ 标记 worker is_busy = false
  │     │           │
  │     │           └─ OnWorkerIdle() → 处理下一个 task 或归还 Worker
  │     │               │
  │     │               ├─ [有更多 task] → PushNormalTask() 继续
  │     │               └─ [无更多 task] → RayletClient::ReturnWorkerLease()
  │     │                   gRPC → NodeManagerService::ReturnWorkerLease
  │     │
  │     └─ 3e. [Task 失败/重试]
  │           ├─ TaskManager::FailOrRetryPendingTask()
  │           └─ [可重试] → ResubmitTask() → 重新进入 3a 流程
  │           └─ [不可重试] → CompletePendingTask() 标记为 failed
```

### 2.2 SchedulingKey 与队列组织

Normal Task 按 **SchedulingKey** 组织队列，SchedulingKey 由以下维度构成：

```
SchedulingKey = (
    SchedulingClass,      // 资源形态编码 (CPU/GPU 数量等)
    FunctionDescriptor,   // 函数描述 (函数名、模块等)
    RuntimeEnvHash        // 运行环境哈希
)
```

每个 SchedulingKey 对应一个 `SchedulingKeyEntry`，包含：

- `task_queue`: 等待 Worker 的 task 队列
- `pending_lease_requests`: 正在等待的 Lease 请求
- `active_workers`: 已获取的 Worker 数量
- `busy_workers`: 正在执行 task 的 Worker 数量

这种组织方式使得相同资源需求的 task 共享 Worker Lease，实现 **Worker 复用**：一个 Worker 执行完一个 task 后，可以直接接收下一个相同 SchedulingKey 的 task，无需归还再重新申请。

### 2.3 Lease Request 节流

`LeaseRequestRateLimiter` 控制 Lease 请求频率，防止过多并发请求压垮 Raylet：

- 基于 `max_pending_lease_requests_per_scheduling_class` 配置
- 超过限制时，不会发送新的 Lease 请求，等待已有请求完成

---

## 3. Actor Task 提交流程

### 3.1 完整调用链

```
CoreWorker::SubmitActorTask()
  │
  ├─ 1. 构建 TaskSpecification (包含 ActorID, 计算序列号)
  │
  ├─ 2. TaskManager::AddPendingTask()
  │
  ├─ 3. ActorTaskSubmitter::SubmitTask(task_spec)
  │     │
  │     ├─ 3a. AddActorQueueIfNotExists(actor_id)
  │     │       └─ 若该 Actor 无队列，创建 ClientQueue
  │     │
  │     ├─ 3b. 检查 Actor 状态 (client_queues_[actor_id].state)
  │     │       │
  │     │       ├─ [DEAD] → 立即 FailPendingTask()
  │     │       └─ [其他] → 继续
  │     │
  │     ├─ 3c. ActorSubmitQueue::Add(task_spec)
  │     │       │ 根据策略选择队列类型:
  │     │       ├─ SequentialActorSubmitQueue: 严格按序列号排队
  │     │       └─ OutOfOrderActorSubmitQueue: 依赖解析完成后即可发送
  │     │
  │     ├─ 3d. LocalDependencyResolver::ResolveDependencies()
  │     │       │ 异步解析依赖
  │     │       ├─ 解析 ObjectRef dependencies
  │     │       ├─ 解析 Actor creation dependencies (等待 Actor 创建完成)
  │     │       └─ 回调: 依赖解析完成 → ActorSubmitQueue::MarkDependenciesResolved()
  │     │
  │     ├─ 3e. [依赖解析完成后] SendPendingTasks()
  │     │       │
  │     │       ├─ 检查 client_queues_[actor_id].state
  │     │       │   ├─ [DEPENDENCIES_UNREADY] → 不发送，等待 ConnectActor()
  │     │       │   ├─ [ALIVE] → 可以发送
  │     │       │   └─ [RESTARTING] → 不发送，等待重连
  │     │       │
  │     │       ├─ [ALIVE] ActorSubmitQueue::PopNextTaskToSend()
  │     │       │   └─ 取出序列号最小且依赖已解析的 task
  │     │       │
  │     │       └─ PushActorTask()
  │     │           │
  │     │           ├─ CoreWorkerClient::PushActorTask()
  │     │           │   gRPC → CoreWorkerService::PushTask
  │     │           └─ 注册 inflight_task_callbacks_[task_attempt]
  │     │
  │     ├─ 3f. [PushTask Reply 回来后]
  │     │       │
  │     │       └─ HandlePushTaskReply()
  │     │           │
  │     │           ├─ [成功] →
  │     │           │   ├─ TaskManager::CompletePendingTask()
  │     │           │   ├─ ActorSubmitQueue::OnTaskDone()
  │     │           │   └─ SendPendingTasks() → 发送下一个 task
  │     │           │
  │     │           ├─ [网络错误/Actor 不可达] →
  │     │           │   ├─ DisconnectActor()
  │     │           │   │   ├─ state = RESTARTING (如果 Actor 可重启)
  │     │           │   │   └─ state = DEAD (如果 Actor 不可重启)
  │     │           │   ├─ 将 in-flight tasks 移入 wait_for_death_info_tasks_
  │     │           │   └─ CheckTimeoutTasks() 等待 GCS 死亡通知
  │     │           │
  │     │           └─ [应用错误] →
  │     │               ├─ TaskManager::FailOrRetryPendingTask()
  │     │               └─ ActorSubmitQueue::OnTaskDone()
  │     │
  │     └─ 3g. [Actor 重连] ConnectActor()
  │           │ 由 GCS 通知触发
  │           ├─ 更新 RPC 地址
  │           ├─ state = ALIVE
  │           ├─ 清理 wait_for_death_info_tasks_ (失败或重新提交)
  │           └─ SendPendingTasks() → 发送积压的 tasks
```

### 3.2 Actor 状态机

每个 Actor 在 `ClientQueue` 中维护独立的状态：

```
    ┌───────────────────────┐
    │  DEPENDENCIES_UNREADY │ ← 初始状态 (Actor 还未连接)
    └───────────┬───────────┘
                │ ConnectActor() 被调用
                ▼
    ┌───────────────────────┐
    │        ALIVE          │ ← Actor 正常运行
    └───────────┬───────────┘
                │ Actor 失败 (网络断开)
                ▼
    ┌───────────────────────┐
    │      RESTARTING       │ ← Actor 正在重启 (仅当 max_restarts > 0)
    └───────────┬───────────┘
                │ Actor 重连成功            │ Actor 永久死亡 / 重启超限
                ▼                          ▼
    ┌───────────────────────┐    ┌───────────────────────┐
    │        ALIVE          │    │         DEAD          │
    └───────────────────────┘    └───────────────────────┘
                                  → 所有 pending tasks 失败
```

### 3.3 顺序 vs 乱序 Actor Task 队列

Ray 支持两种 Actor Task 执行顺序策略：

**SequentialActorSubmitQueue**（默认）：
- Task 严格按提交顺序（序列号）执行
- `PopNextTaskToSend()` 返回序列号最小的 task
- 即使后续 task 依赖已解析，也必须等待前面的 task 完成
- 保证 Actor 状态按确定性顺序变更

**OutOfOrderActorSubmitQueue**：
- Task 按依赖解析完成顺序执行
- `PopNextTaskToSend()` 返回第一个依赖已解析的 task
- 无数据依赖的 task 可以跳过前面的 task 先执行
- 提高并发度，但牺牲确定性状态变更顺序

### 3.4 Actor Creation Task 流程

Actor 创建任务走独立路径，直接与 GCS 交互：

```
ActorTaskSubmitter::SubmitActorCreationTask()
  │
  ├─ ActorCreator::AsyncCreateActor()
  │   gRPC → ActorInfoGcsService::CreateActor
  │
  ├─ [GCS 创建 Actor] →
  │   ├─ 选择目标 Node
  │   ├─ 在目标 Node 上启动 Actor Worker
  │   └─ 返回 Actor 地址给所有依赖此 Actor 的 CoreWorker
  │
  └─ [CoreWorker 收到 Actor 地址通知]
      └─ ConnectActor() → 建立 RPC 连接 → 发送 pending tasks
```

---

## 4. Dependency Resolution 机制

### 4.1 LocalDependencyResolver 架构

`LocalDependencyResolver` 负责在 task 提交前解析所有参数依赖：

```
LocalDependencyResolver::ResolveDependencies(task_spec, callback)
  │
  ├─ 创建 TaskState (追踪解析进度)
  │
  ├─ 遍历 task_spec 的所有 args:
  │   │
  │   ├─ [ObjectRef arg]
  │   │   ├─ [Object 在 in_memory_store_ 中]
  │   │   │   └─ 直接 inline: 将 Object 数据嵌入 task_spec
  │   │   │   └─ obj_dependencies_remaining-- (本地依赖立即完成)
  │   │   │
  │   │   └─ [Object 不在本地]
  │   │       └─ 保留 ObjectRef 引用 (remote dependency)
  │   │       └─ obj_dependencies_remaining 保持 (在 Worker 侧解析)
  │   │
  │   └─ [ActorHandle arg]
  │     └─ actor_dependencies_remaining++
  │     └─ 等待 Actor 创建完成
  │
  ├─ [所有本地依赖已解析] →
  │   ├─ actor_dependencies_remaining == 0
  │   └─ obj_dependencies_remaining == 0
  │   └─ 立即调用 callback(task_spec, Status::OK())
  │
  └─ [有未完成依赖] →
  │   ├─ 注册依赖等待回调
  │   ├─ [Actor 创建完成] → actor_dependencies_remaining-- → CheckComplete()
  │   └─ [所有依赖完成] → callback(task_spec, Status::OK())
  │
  └─ [依赖失败] →
      └─ callback(task_spec, failure_status)
```

### 4.2 依赖解析的关键决策

| 依赖类型 | 解析策略 | 说明 |
|---------|---------|------|
| Local Object (在 in_memory_store_) | Inline | 将数据直接嵌入 task spec，减少网络传输 |
| Remote Object (在 Plasma Store) | 引用传递 | 仅传递 ObjectRef，由执行侧 Worker 获取 |
| Actor Handle | 等待创建 | 确保目标 Actor 已创建后再提交 |
| Nested Task Return | 传递 ObjectRef | 生成器返回值用 ObjectRefStream 管理 |

### 4.3 CancelDependencyResolution

当 task 被取消时，需要取消正在进行的依赖解析：

```
LocalDependencyResolver::CancelDependencyResolution(task_id)
  │
  ├─ 从 tasks_ map 中移除 TaskState
  ├─ 取消所有注册的依赖等待回调
  └─ 释放相关资源引用计数
```

---

## 5. Task Manager 生命周期管理

### 5.1 Task 状态流转

```
                    ┌─────────────────┐
                    │  PENDING_TASK   │ ← AddPendingTask()
                    │  (等待执行)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌──────────┐ ┌───────────────┐
    │ COMPLETE_TASK   │ │ FAILED   │ │ RETRY_TASK    │
    │ (成功完成)      │ │ (失败)   │ │ (重试)        │
    └─────────────────┘ └──────────┘ └───────┬───────┘
                                         │
                                         ▼
                                   ┌─────────────────┐
                                   │  PENDING_TASK   │
                                   │  (重新等待)      │
                                   └─────────────────┘
```

### 5.2 TaskManager 核心方法

**AddPendingTask()**:
- 注册 task 为 pending 状态
- 创建 return ObjectRef（由 ReferenceCounter 管理）
- 记录 task spec、max_retries、caller 地址等
- 如果是生成器 task，创建 ObjectRefStream

**CompletePendingTask()**:
- 标记 task 为完成状态
- 处理 return objects（存储结果到 Plasma 或 inline）
- 通知 ReferenceCounter 更新引用计数
- 清理 task 相关的中间状态

**FailOrRetryPendingTask()**:
- 根据 max_retries 和错误类型决定重试或失败
- 重试时：ResubmitTask() → 重新进入提交流程
- 不重试时：CompletePendingTask() 标记为 failed
- 处理 ObjectRefStream 的关闭（生成器 task 失败）

**OnTaskDependenciesInlined()**:
- 依赖被 inline 到 task spec 后，释放原始 ObjectRef 的引用计数
- 配合 LocalDependencyResolver 的 inline 机制

### 5.3 ObjectRefStream（生成器 Task）

对于返回多个值的生成器 task，TaskManager 使用 `ObjectRefStream` 管理：

```
ObjectRefStream
  │
  ├─ 维护一个有序的 ObjectRef 队列
  ├─ 每次 generator yield → 新增一个 ObjectRef
  ├─ 前端读取 → 从队列头部取 ObjectRef
  └─ generator 完成 → 关闭 stream，标记 end_of_stream
```

---

## 6. Raylet 侧调度与执行

### 6.1 Worker Lease 授予流程

```
RayletClient::RequestWorkerLease()
  │ gRPC → NodeManagerService::RequestWorkerLease
  ▼
NodeManager::HandleRequestWorkerLease(request, reply, callback)
  │
  ├─ 1. 检查是否已处理 (防重复)
  │
  ├─ 2. 验证调用者存活 (非 detached 场景)
  │
  ├─ 3. 检查是否已在 cluster/local lease manager 中排队
  │
  └─ 4. ClusterLeaseManager::QueueAndScheduleLease()
      │
      ├─ ClusterResourceScheduler::GetBestSchedulableNode()
      │   │ 考虑: 资源可用性、依赖本地性、spread策略
      │   │
      │   ├─ [最优节点 = 本节点] →
      │   │   └─ LocalLeaseManager::QueueAndScheduleLease()
      │   │       │
      │   │       ├─ LeaseDependencyManager::WaitForLeaseArgsRequests()
      │   │       │   │ 检查 task 依赖的 Object 是否在本节点 Plasma Store
      │   │       │   ├─ [依赖齐全] → 继续
      │   │       │   └─ [依赖缺失] → 注册等待，等依赖拉取到本节点
      │   │       │
      │   │       ├─ ClusterResourceScheduler::AllocateLocalTaskResources()
      │   │       │   │ 在本节点分配 CPU/GPU 等资源
      │   │       │   ├─ [资源充足] → 继续
      │   │       │   └─ [资源不足] → 等待资源释放
      │   │       │
      │   │       ├─ WorkerPool::PopWorker()
      │   │       │   │
      │   │       │   ├─ FindAndPopIdleWorker()
      │   │       │   │   └─ 找到匹配的 idle worker
      │   │       │   │
      │   │       │   └─ [无 idle worker] → StartNewWorker()
      │   │       │       └─ 启动新 Worker 进程
      │   │       │
      │   │       └─ LocalLeaseManager::Grant(worker, lease)
      │   │           │
      │   │           ├─ worker->GrantLease(lease)
      │   │           ├─ 设置 reply: worker IP/port/ID/node_id
      │   │           ├─ 设置 reply: resource mapping
      │   │           └─ send_reply_callback() → 回复给 CoreWorker
      │   │
      │   └─ [最优节点 = 远程节点] →
      │       └─ Lease Spill: 将 lease 转发到目标节点的 raylet
      │           └─ 目标 raylet 执行类似的本地调度流程
```

### 6.2 Worker Lease 归还流程

```
CoreWorker → RayletClient::ReturnWorkerLease()
  │ gRPC → NodeManagerService::ReturnWorkerLease
  ▼
NodeManager::HandleReturnWorkerLease(request, reply, callback)
  │
  ├─ 1. 从 leased_workers_ 中找到 worker
  │
  ├─ 2. ReleaseWorker(lease_id)
  │   └─ 从 leased_workers_ map 中移除
  │
  ├─ 3. LocalLeaseManager::ReleaseWorkerResources(worker)
  │   └─ 释放分配给该 worker 的 CPU/GPU 等资源
  │   └─ 更新 ClusterResourceScheduler 的本地资源视图
  │
  └─ 4. [Worker 不应退出] → HandleWorkerAvailable(worker)
      └─ 将 worker 放回 idle pool，等待下一个 lease
```

---

## 7. Task Receiver 接收与执行

### 7.1 接收 Task 流程

```
Remote Worker CoreWorker 收到 PushTask RPC
  │ gRPC → CoreWorkerService::PushTask
  ▼
CoreWorker::HandlePushTask(request, reply, callback)
  │
  ├─ 1. 解析 PushTaskRequest
  │   ├─ TaskSpecification
  │   ├─ 依赖 ObjectRefs (需在本地解析)
  │   └─ 序列号 (Actor Task)
  │
  ├─ 2. TaskReceiver::HandleTask()
  │   │
  │   ├─ [Normal Task] →
  │   │   └─ scheduling_queue_.AddNormalTask(task)
  │   │   └─ RunNormalTasksFromQueue()
  │   │
  │   └─ [Actor Task] →
  │       ├─ scheduling_queue_.AddActorTask(task)
  │       └─ 按 concurrency group 分配执行
  │
  ├─ 3. [等待依赖] DependencyWaiter::Wait()
  │   │ 等待 remote ObjectRefs 在本地 Plasma Store 中可用
  │   └─ [依赖可用] → 继续执行
  │
  ├─ 4. 调用 task_handler_ (语言前端回调)
  │   │ Python: 执行用户函数
  │   │ Java: 执行用户方法
  │   └─ 返回执行结果
  │
  ├─ 5. 构造 PushTaskReply
  │   ├─ return_objects (执行结果)
  │   ├─ borrowed_refs (跨 Worker 引用信息)
  │   └─ status
  │
  └─ 6. send_reply_callback() → 回复给提交方 CoreWorker
```

### 7.2 SchedulingQueue 与执行顺序

```
SchedulingQueue
  │
  ├─ normal_tasks_ 队列
  │   └─ RunNormalTasksFromQueue() 顺序执行
  │
  ├─ actor_tasks_ 队列
  │   └─ 按 concurrency group 管理
  │
  ├─ ConcurrencyGroupManager
  │   ├─ default_group: Actor 主线程任务
  │   └─ named_groups: 并发执行组 (如 IO 组)
  │   └─ 每个 group 有独立执行队列和线程
  │
  └─ 优先级: actor_tasks > normal_tasks
```

---

## 8. 关键数据结构

### 8.1 NormalTaskSubmitter

```cpp
// 按 SchedulingKey 组织的 task 队列
struct SchedulingKeyEntry {
    std::deque<TaskSpecification> task_queue;        // 等待执行的 task
    absl::flat_hash_map<LeaseID, rpc::Address> pending_lease_requests;  // 正在请求的 lease
    int active_workers;                               // 已获得的 worker 数
    int busy_workers;                                 // 正在执行 task 的 worker 数
};

absl::flat_hash_map<SchedulingKey, SchedulingKeyEntry> scheduling_key_entries_;

// Worker lease 跟踪
struct LeaseEntry {
    bool is_busy;                 // 是否正在执行 task
    SchedulingKey scheduling_key; // worker 对应的资源类型
    TaskID executing_task_id;     // 当前执行的 task
};

absl::flat_hash_map<rpc::Address, LeaseEntry> worker_to_lease_entry_;
```

### 8.2 ActorTaskSubmitter

```cpp
// 每个 Actor 的状态和队列
struct ClientQueue {
    rpc::Address actor_address;                    // Actor 的 RPC 地址
    ActorState state;                              // DEPENDENCIES_UNREADY / ALIVE / RESTARTING / DEAD
    std::unique_ptr<IActorSubmitQueue> actor_submit_queue_;  // 顺序/乱序队列
    int64_t next_task_sequence_number;             // 下一个序列号
};

absl::flat_hash_map<ActorID, ClientQueue> client_queues_;

// 等待死亡信息的 task
struct PendingTaskWaitingForDeathInfo {
    TaskSpecification task_spec;
    TaskAttempt task_attempt;
    rpc::ClientCallback<rpc::PushTaskReply> callback;
    absl::Time start_time;
};

std::deque<std::shared_ptr<PendingTaskWaitingForDeathInfo>> wait_for_death_info_tasks_;

// 在途 task 的回调
absl::flat_hash_map<TaskAttempt, rpc::ClientCallback<rpc::PushTaskReply>> inflight_task_callbacks_;
```

### 8.3 TaskManager

```cpp
// Task 跟踪条目
struct TaskEntry {
    TaskSpecification task_spec;
    int64_t max_retries;
    int64_t num_retries_left;
    TaskStatus status;
    ObjectRefStream object_ref_stream;  // 仅用于生成器 task
};

absl::flat_hash_map<TaskID, std::shared_ptr<TaskEntry>> submissible_tasks_;
```

---

## 9. 错误处理与重试机制

### 9.1 Normal Task 错误处理

```
[Worker 死亡 / Task 执行失败]
  │
  ├─ RayletClient::GetWorkerFailureCause()
  │   └─ 查询失败原因 (系统错误 vs 应用错误)
  │
  ├─ [应用错误] →
  │   └─ TaskManager::FailOrRetryPendingTask()
  │       ├─ [有重试次数] → ResubmitTask() → 重新提交
  │       └─ [无重试次数] → CompletePendingTask(failed)
  │
  ├─ [系统错误 / Worker 死亡] →
  │   └─ 正常重试流程 (max_retries 控制重试上限)
  │
  └─ [Lease 请求失败] →
      ├─ 尝试 fallback 到本地 raylet
      └─ 或标记 task 为 failed
```

### 9.2 Actor Task 错误处理

```
[Actor 网络断开]
  │
  ├─ DisconnectActor()
  │   ├─ [max_restarts > 0] → state = RESTARTING
  │   │   ├─ 所有 in-flight tasks 移入 wait_for_death_info_tasks_
  │   │   ├─ 断开 RPC 连接
  │   │   └─ 等待 GCS 的 Actor 死亡/重启通知
  │   │
  │   └─ [max_restarts == 0 或已超限] → state = DEAD
  │       └─ 所有 pending + in-flight tasks → FailPendingTask()
  │
  ├─ [收到 GCS Actor 重启通知]
  │   └─ ActorCreator::AsyncRestartActorForLineageReconstruction()
  │       └─ Actor 重启后 → ConnectActor() → 重新发送 pending tasks
  │
  ├─ [超时等待死亡信息] CheckTimeoutTasks()
  │   └─ 超时后 → 假定 Actor 已死亡 → FailPendingTask()
  │
  └─ [依赖解析失败]
      └─ MarkDependencyFailed() → FailOrRetryPendingTask()
```

### 9.3 Task 取消流程

```
CoreWorker::CancelTask(task_id, force_kill)
  │
  ├─ [Normal Task] NormalTaskSubmitter::CancelTask()
  │   ├─ 加入 cancelled_tasks_ set
  │   ├─ [task 在 task_queue] → 直接移除
  │   ├─ [task 在 pending_lease] → RayletClient::CancelWorkerLease()
  │   └─ [task 在执行中] → RayletClient::CancelLocalTask()
  │
  ├─ [Actor Task] ActorTaskSubmitter::CancelTask()
  │   ├─ 标记 task 为 cancelled
  │   ├─ [task 在队列] → 从 ActorSubmitQueue 移除
  │   └─ [task 在执行中] → CoreWorkerClient::RequestOwnerToCancelTask()
```

---

## 10. gRPC 接口定义

### 10.1 Core Worker Service

| Service | Method | Request | Reply | 说明 |
|---------|--------|---------|-------|------|
| `CoreWorkerService` | `PushTask` | `PushTaskRequest` | `PushTaskReply` | 推送 Normal/Actor Task 到远程 Worker |
| `CoreWorkerService` | `KillActor` | `KillActorRequest` | `KillActorReply` | 强制杀死 Actor |

### 10.2 Node Manager Service (Raylet)

| Service | Method | Request | Reply | 说明 |
|---------|--------|---------|-------|------|
| `NodeManagerService` | `RequestWorkerLease` | `RequestWorkerLeaseRequest` | `RequestWorkerLeaseReply` | 请求租用 Worker |
| `NodeManagerService` | `ReturnWorkerLease` | `ReturnWorkerLeaseRequest` | `ReturnWorkerLeaseReply` | 归还租用的 Worker |
| `NodeManagerService` | `CancelWorkerLease` | `CancelWorkerLeaseRequest` | `CancelWorkerLeaseReply` | 取消 Worker Lease 请求 |
| `NodeManagerService` | `GetWorkerFailureCause` | `GetWorkerFailureCauseRequest` | `GetWorkerFailureCauseReply` | 获取 Worker 失败原因 |

### 10.3 GCS Actor Service

| Service | Method | Request | Reply | 说明 |
|---------|--------|---------|-------|------|
| `ActorInfoGcsService` | `CreateActor` | `CreateActorRequest` | `CreateActorReply` | 创建 Actor |
| `ActorInfoGcsService` | `RestartActor` | `RestartActorRequest` | `RestartActorReply` | 重启 Actor |

---

## 附录：完整端到端流程图

### Normal Task 端到端

```
[Python Frontend]         [CoreWorker (Submitter)]            [Raylet]              [Remote Worker]
     │                          │                               │                        │
     │ f.remote(*args)          │                               │                        │
     ├──────────────────────►   │                               │                        │
     │                          │ SubmitTask()                  │                        │
     │                          ├─ AddPendingTask()             │                        │
     │                          ├─ ResolveDependencies()        │                        │
     │                          │  (inline local args)          │                        │
     │                          │                               │                        │
     │                          │ [dep resolved]                │                        │
     │                          ├─ Queue by SchedulingKey       │                        │
     │                          ├─ RequestWorkerLease() ──────► │                        │
     │                          │                               │ QueueAndSchedule()     │
     │                          │                               │ AllocateResources()    │
     │                          │                               │ PopWorker()            │
     │                          │                               │                        │
     │                          │◄─ Lease Granted ──────────────┤                        │
     │                          │  (worker_address)             │                        │
     │                          │                               │                        │
     │                          │ PushNormalTask() ─────────────────────────────────────► │
     │                          │                               │                        │ HandleTask()
     │                          │                               │                        │ WaitDependencies()
     │                          │                               │                        │ Execute()
     │                          │                               │                        │
     │                          │◄─ PushTaskReply ───────────────────────────────────────┤
     │                          │  (results)                    │                        │
     │                          │                               │                        │
     │                          ├─ CompletePendingTask()        │                        │
     │                          │                               │                        │
     │                          │ [more tasks?]                 │                        │
     │                          ├─ Yes: Push next task          │                        │
     │                          ├─ No: ReturnWorkerLease() ──► │                        │
     │                          │                               │ ReleaseResources()     │
     │                          │                               │ WorkerAvailable()      │
     │                          │                               │                        │
     │◄─ Return ObjectRef ─────┤                               │                        │
     │                          │                               │                        │
```

### Actor Task 端到端

```
[Python Frontend]    [CoreWorker (Submitter)]     [GCS]       [Raylet]    [Actor Worker]
     │                     │                        │            │              │
     │ actor.method()      │                        │            │              │
     ├──────────────────►  │                        │            │              │
     │                     │ SubmitActorTask()      │            │              │
     │                     ├─ AddPendingTask()      │            │              │
     │                     ├─ AddActorQueue()       │            │              │
     │                     ├─ ResolveDependencies() │            │              │
     │                     ├─ Queue by sequence_no  │            │              │
     │                     │                        │            │              │
     │                     │ [actor alive?]         │            │              │
     │                     ├─ Yes: SendPendingTasks │            │              │
     │                     ├─ No: Wait for Connect  │            │              │
     │                     │                        │            │              │
     │                     │ PushActorTask() ──────────────────────────────────► │
     │                     │                        │            │              │ HandleTask()
     │                     │                        │            │              │ Execute()
     │                     │                        │            │              │
     │                     │◄─ PushTaskReply ───────────────────────────────────┤
     │                     │  (results)             │            │              │
     │                     │                        │            │              │
     │                     ├─ CompletePendingTask() │            │              │
     │                     ├─ OnTaskDone() → Send next task     │              │
     │                     │                        │            │              │
     │◄─ Return ObjectRef ─┤                        │            │              │
     │                     │                        │            │              │

     │                     │                        │            │              │
     │ [Actor fails]       │                        │            │              │ ✗
     │                     │◄─ RPC error            │            │              │
     │                     │ DisconnectActor()      │            │              │
     │                     │ state = RESTARTING     │            │              │
     │                     │                        │            │              │
     │                     │◄─ GCS actor restart notification ──┤              │
     │                     │                        │            │              │
     │                     │ ConnectActor()         │            │              │ [new actor]
     │                     │ state = ALIVE          │            │              │
     │                     │ SendPendingTasks() ───────────────────────────────► │
     │                     │                        │            │              │
```

---

## 关键源码文件索引

| 文件 | 职责 |
|------|------|
| [normal_task_submitter.h](../src/ray/core_worker/task_submission/normal_task_submitter.h) | Normal Task 提交接口 |
| [normal_task_submitter.cc](../src/ray/core_worker/task_submission/normal_task_submitter.cc) | Normal Task 提交实现 |
| [actor_task_submitter.h](../src/ray/core_worker/task_submission/actor_task_submitter.h) | Actor Task 提交接口 |
| [actor_task_submitter.cc](../src/ray/core_worker/task_submission/actor_task_submitter.cc) | Actor Task 提交实现 |
| [dependency_resolver.h](../src/ray/core_worker/task_submission/dependency_resolver.h) | 依赖解析接口 |
| [dependency_resolver.cc](../src/ray/core_worker/task_submission/dependency_resolver.cc) | 依赖解析实现 |
| [task_manager.h](../src/ray/core_worker/task_manager.h) | Task 生命周期管理接口 |
| [task_manager.cc](../src/ray/core_worker/task_manager.cc) | Task 生命周期管理实现 |
| [task_receiver.h](../src/ray/core_worker/task_execution/task_receiver.h) | Task 接收接口 |
| [task_receiver.cc](../src/ray/core_worker/task_execution/task_receiver.cc) | Task 接收实现 |
| [core_worker.cc](../src/ray/core_worker/core_worker.cc) | CoreWorker 主实现 |
| [node_manager.cc](../src/ray/raylet/node_manager.cc) | Raylet NodeManager |
| [cluster_lease_manager.cc](../src/ray/raylet/scheduling/cluster_lease_manager.cc) | 分布式 Lease 调度 |
| [local_lease_manager.cc](../src/ray/raylet/scheduling/local_lease_manager.cc) | 本地 Lease 管理 |
| [worker_pool.cc](../src/ray/raylet/worker_pool.cc) | Worker 池管理 |