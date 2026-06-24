# Ray Dashboard 数据流全解析

> 基于 Ray 2.53.0 源码分析，梳理 Dashboard 从数据采集到前端渲染的完整链路。

---

## 一、架构总览

Dashboard 采用 **中心化采集 + 分层组织 + API 暴露** 的三层架构：

```
┌─────────────────────────────────────────────────────────┐
│  Ray Core (GCS / Raylet / Reporter Agent / Worker)      │
│  数据源头：ActorTable, NodeTable, TaskTable, ResourceUsage... │
└──────────────────────┬──────────────────────────────────┘
                       │ GCS PubSub / gRPC / HTTP Agent Report
                       ▼
┌─────────────────────────────────────────────────────────┐
│  node_head.py (DashboardHead)                           │
│  _update_nodes(), _update_actors(), _update_node_stats()│
│  _update_node_physical_stats()                          │
│  写入 DataSource（静态字典，全局共享）                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  DataOrganizer (datacenter.py)                          │
│  organize(): 合并 node_stats + node_physical_stats → workers│
│  get_node_info(), get_actor_infos() 等聚合方法           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  API Endpoints                                          │
│  node_head.py: /nodes, /logical/actors, /logical/actors/{id}│
│  state_head.py: /api/v0/tasks, /api/v0/workers, /api/v0/logs ...│
│  event_head.py: /events, /api/v0/cluster_events          │
│  job_head.py: /api/jobs/                                 │
│  serve_head.py: /api/serve/applications/                  │
│  其他模块各自暴露 API                                     │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP REST
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Frontend (React + SWR)                                 │
│  service/*.ts → API 请求                                  │
│  hook/*.ts → SWR 缓存 + 定时刷新                          │
│  pages/* → 页面组件渲染                                   │
└─────────────────────────────────────────────────────────┘
```

### 关键设计点

1. **DataSource 是全局静态字典**：所有采集数据直接写入 `DataSource` 类属性（类级别 dict），供各模块共享读取。
2. **两套 API 体系并存**：
   - **Legacy API**（`/logical/actors`、`/nodes` 等）→ 由 `node_head.py` 提供，数据来自 `DataSource + DataOrganizer`
   - **State API**（`/api/v0/*`）→ 由 `state_head.py` 提供，数据来自 `StateAPIManager + StateDataSourceClient`（直接查 GCS）
3. **GCS PubSub 防止 TOCTOU**：所有订阅者先 subscribe 再 get-all，确保增量更新不遗漏。
4. **DataOrganizer.purge() 定期清理**：只保留 ALIVE 节点的数据，过期节点数据被清除。

---

## 二、DataSource 字典一览

| 属性 | 类型 | 含义 | 数据来源 |
|------|------|------|----------|
| `node_stats` | `{node_id: GetNodeStatsReply dict}` | Raylet 运行时统计 | gRPC → Raylet NodeManager |
| `node_physical_stats` | `{node_id: reporter dict}` | 物理资源使用（CPU/GPU/MEM/网络） | GCS PubSub → Reporter Agent |
| `actors` | `{actor_id: ActorTableData dict}` | Actor 全量信息 | GCS PubSub → GcsAioActorSubscriber |
| `nodes` | `{node_id: GcsNodeInfo dict}` | 集群节点信息 | GCS PubSub → GcsAioNodeInfoSubscriber |
| `node_workers` | `{node_id: worker_list}` | 节点上的 Worker 列表（合并后） | DataOrganizer.organize() 产出 |
| `node_actors` | `{node_id: {actor_id: ActorTableData}}` | 按节点分组的 Actor | _update_actors() 派生 |
| `core_worker_stats` | `{worker_id: core_worker_stats}` | Core Worker 统计 | DataOrganizer.organize() 产出 |

> `node_workers` 和 `core_worker_stats` 不是直接采集的，而是 `DataOrganizer.organize()` 合并 `node_stats.coreWorkersStats` + `node_physical_stats.workers` 后的产物。

---

## 三、各数据类型详细链路

### 3.1 Actor 数据流

这是最复杂的数据流，涉及多数据源聚合。

```
GCS ActorTable
  │
  │ GcsAioActorSubscriber.subscribe() + poll(batch_size=200)
  ▼
node_head.py._update_actors()                [L551-618]
  │ 先 subscribe → 再 get-all（防 TOCTOU）
  │ _get_all_actors() → DataSource.actors = actor_dicts
  │ 增量 → _process_updated_actor_table() → DataSource.actors / DataSource.node_actors
  ▼
DataOrganizer.get_actor_infos()              [L186-198]
  │ 查 DataSource.actors → 调 _get_actor_info() 聚合
  │   聚合 core_worker_stats (进程信息)
  │   聚合 node_physical_stats.workers (进程统计/CPU/GPU)
  │   聚合 node_physical_stats.gpus (GPU 利用率)
  │   聚合 node_physical_stats.mem (内存)
  │   转换 requiredResources 格式
  ▼
node_head.py: GET /logical/actors/{actor_id} [L715-724]
  │  → DataOrganizer.get_actor_infos(actor_ids=[actor_id])
  ▼
前端: getActor(actorId) → GET logical/actors/${actorId}
  │  → service/actor.ts [L24-26]
  ▼
SWR Hook: useActorDetail()                   [hook/useActorDetail.ts]
  │ refreshInterval = API_REFRESH_INTERVAL_MS
  ▼
ActorDetailPage                              [pages/actor/ActorDetail.tsx]
  展示: state, actorId, name, className, jobId, nodeId, pid,
        startTime, endTime, restartCount, exitDetails, requiredResources,
        gpus, processStats, mem, logs...
```

**前端路由**：
- `/actors` → `Actors` 列表页
- `/actors/:actorId` → `ActorDetailLayout` → `ActorDetailPage`
- `/actors/:actorId/tasks/:taskId` → 嵌入的 Task 详情页

**前端关键文件**：
- [pages/actor/ActorDetail.tsx](../python/ray/dashboard/client/src/pages/actor/ActorDetail.tsx) — 详情页组件
- [pages/actor/ActorLayout.tsx](../python/ray/dashboard/client/src/pages/actor/ActorLayout.tsx) — 布局框架
- [pages/actor/hook/useActorDetail.ts](../python/ray/dashboard/client/src/pages/actor/hook/useActorDetail.ts) — SWR 数据 hook
- [service/actor.ts](../python/ray/dashboard/client/src/service/actor.ts) — API service
- [type/actor.ts](../python/ray/dashboard/client/src/type/actor.ts) — TS 类型定义
- [pages/actor/ActorLogs.tsx](../python/ray/dashboard/client/src/pages/actor/ActorLogs.tsx) — Actor 日志组件

**后端关键文件**：
- [modules/node/node_head.py](../python/ray/dashboard/modules/node/node_head.py) — `_update_actors()` L551, `get_actor()` L715
- [modules/node/datacenter.py](../python/ray/dashboard/modules/node/datacenter.py) — `get_actor_infos()` L186, `_get_actor_info()` L200

**State API 也提供 Actor**：
- `GET /api/v0/actors` → state_head.py，直接从 GCS StateDataSourceClient 查询
- `GET /api/v0/actors/summarize` → Actor 统计摘要

---

### 3.2 Node 数据流

```
GCS NodeTable
  │
  │ GcsAioNodeInfoSubscriber
  ▼
node_head.py._update_nodes()                 [L298+]
  │ subscribe → get-all → 增量更新
  │ DataSource.nodes = node_dicts
  │ 同时为每个 ALIVE 节点创建 gRPC stub (NodeManagerService)
  ▼
DataOrganizer.get_node_info(node_id)         [L131-174]
  │ 合并 node_physical_stats + node_stats + GcsNodeInfo
  │ 添加 object_store 统计 (used/avail memory)
  │ 添加 stateMessage (deathInfo)
  │ 若非 summary: 合并 node_actors + node_workers
  ▼
node_head.py: GET /nodes                     [L385] → get_all_node_summary()
node_head.py: GET /nodes/{node_id}           [L419] → get_node_info(node_id)
  ▼
前端: getNodeList() → GET nodes?view=summary
前端: getNodeDetail(id) → GET nodes/${id}
  │  → service/node.ts
  ▼
NodeDetailPage                               [pages/node/]
  展示: 节点状态、CPU/GPU/MEM/磁盘、Object Store、Workers 列表、Actors 列表...
```

**前端路由**：
- `/cluster` → `Nodes` 列表页
- `/cluster/nodes/:id` → `NodeDetailPage`

---

### 3.3 Worker 数据流

Worker 数据不单独有页面，嵌在 Node 详情页中。

```
Raylet NodeManager (gRPC GetNodeStats)
  │
  │ node_head.py._update_node_stats()         [L430-521]
  │   定期遍历所有 ALIVE 节点，调用 GetNodeStats gRPC
  │   DataSource.node_stats[node_id] = stats_reply
  ▼
Reporter Agent (via GCS PubSub)
  │
  │ node_head.py._update_node_physical_stats() [~L520+]
  │   GcsAioResourceUsageSubscriber
  │   DataSource.node_physical_stats[node_id] = physical_stats
  ▼
DataOrganizer.organize()                     [L59-98]
  │ 遍历 DataSource.nodes
  │   合并 node_physical_stats.workers + node_stats.coreWorkersStats
  │   _extract_workers_for_node():
  │     按 PID 匹配 core_worker_stats 到 physical_stats.worker
  │     添加 language, jobId, coreWorkerStats 字段
  │   → DataSource.node_workers[node_id] = workers
  │   → DataSource.core_worker_stats[worker_id] = stats
  ▼
State API: GET /api/v0/workers               [state_head.py L138]
  │ 直接从 GCS 查询
  ▼
前端: 在 NodeDetailPage 的 Workers 表格中展示
```

**关键合并逻辑**（`_extract_workers_for_node`）：
- Reporter Agent 提供物理统计（PID、CPU、内存）
- Raylet 提供逻辑统计（language, jobId, coreWorkerStats）
- 两者通过 PID 关联合并

---

### 3.4 Task 数据流

Task 数据完全通过 State API 获取，不走 DataSource/DataOrganizer 路径。

```
GCS TaskTable
  │
  │ StateDataSourceClient → StateAPIManager
  ▼
state_head.py: GET /api/v0/tasks              [L144]
  │ 参数: detail, limit, filter_keys, filter_predicates, filter_values
  │   例: filter_keys=job_id&filter_predicates==&filter_values=xxx
  ▼
前端: getTasks(jobId) → GET api/v0/tasks?detail=1&limit=10000&...
前端: getTask(taskId) → GET api/v0/tasks?detail=1&limit=1&filter_keys=task_id...
  │  → service/task.ts
  ▼
TaskList / TaskDetailPage                     [pages/state/task.tsx]
  展示: task状态、耗时、函数名、参数、返回值...
```

**其他 Task 端点**：
- `GET /api/v0/tasks/summarize` → Task 统计摘要
- `GET /api/v0/tasks/timeline` → Task 时间线（用于 Timeline 视图）

---

### 3.5 Job 数据流

Job 有两套 API：

```
┌── Legacy Job API (job_head.py) ──────────────────┐
│                                                    │
│  GET /api/jobs/            → job 列表              │
│  GET /api/jobs/{id}        → job 详情              │
│  POST /api/jobs/{id}/stop  → 停止 job              │
│                                                    │
│  数据来源: GCS JobTable + DashboardHead 收集         │
│  前端: service/job.ts → getJobList(), getJobDetail()│
│  页面: /jobs → JobList, /jobs/:id → JobDetail      │
└────────────────────────────────────────────────────┘

┌── State API (state_head.py) ─────────────────────┐
│                                                    │
│  GET /api/v0/jobs           → job 列表             │
│                                                    │
│  数据来源: StateDataSourceClient → GCS              │
└────────────────────────────────────────────────────┘
```

**前端路由**：
- `/jobs` → `JobList` 列表页
- `/jobs/:id` → `JobPage` → `JobDetailLayout` → Job 详情页
- `/jobs/:id/tasks/:taskId` → 嵌入的 Task 详情页

---

### 3.6 Log 数据流

```
各节点上的日志文件
  │
  │ StateDataSourceClient (LogsManager) → 查询各节点 agent
  ▼
state_head.py: GET /api/v0/logs               [L162] → list_logs()
state_head.py: GET /api/v0/logs/{media_type}  [L213] → get_logs() (流式)
  │ media_type 参数指定日志文件路径
  ▼
前端: listStateApiLogs() → GET api/v0/logs
前端: getStateApiLog(filePath) → GET api/v0/logs/file
  │  → service/log.ts
  ▼
StateApiLogsListPage / StateApiLogViewerPage   [pages/log/]
  展示: 日志文件列表、日志内容（流式传输）
```

**Actor 日志的特殊路径**：
- Actor 详情页内嵌 `ActorLogs` 组件
- 也通过 `/api/v0/logs` 获取，但按 actor 的 PID/日志文件名过滤

---

### 3.7 Object 数据流

```
GCS ObjectTable
  │
  │ StateDataSourceClient
  ▼
state_head.py: GET /api/v0/objects             [L150]
state_head.py: GET /api/v0/objects/summarize   → Object 统计摘要
  │
  │ 前端无独立页面，数据在 Job/Task 详情页中引用
  ▼
```

---

### 3.8 Placement Group 数据流

```
GCS PlacementGroupTable
  │
  │ StateDataSourceClient
  ▼
state_head.py: GET /api/v0/placement_groups     [~L155]
  │
  │ 前端: service/placementGroup.ts → getPlacementGroups()
  ▼
PlacementGroupListPage                         [pages/placementGroup/]
```

---

### 3.9 Event 数据流

```
各组件上报事件
  │
  │ POST /report_events → event_head.py 收集
  │ 或从 GCS ClusterEventTable 查询
  ▼
event_head.py:
  GET /events?job_id={id}       → 过滤事件的 pipeline 视图
  GET /api/v0/cluster_events     → 集群事件列表（分页/过滤）
  │
  │ 前端: service/event.ts → getEvents(), getGlobalEvents()
  ▼
事件在 Job 详情页、Actor 详情页等页面中展示
```

---

### 3.10 Serve 数据流

```
Serve Controller Actor (Ray Serve 内部状态)
  │
  │ serve_head.py.get_serve_controller() → 通过 Actor 获取 Serve 状态
  ▼
serve_head.py:
  GET /api/serve/applications/       → Serve 应用列表/详情
  DELETE /api/serve/applications/     → 删除 Serve 应用
  PUT /api/serve/applications/        → 部署/更新 Serve 应用
  POST .../scale                      → 调整 Deployment 副本数
  │
  │ 前端: service/serve.ts → getServeApplications()
  ▼
Serve 页面: /serve → Serve 系统页、Deployments 列表、应用详情
```

---

### 3.11 Train 数据流

```
GCS ActorTable (Train 相关的 Actor)
  │
  │ train_head.py 通过 Actor 查询获取 Train Run 信息
  │ _add_actor_status_and_update_run_status()
  ▼
train_head.py:
  GET /api/train/v2/runs/v1    → Train V2 Run 信息
  GET /api/train/v2/runs       → Train V1 Run 信息
  │
  │ 前端: 在 Train 页面展示
  ▼
```

---

### 3.12 Metrics / Prometheus 数据流

```
Prometheus Server
  │
  │ metrics_head.py._query_prometheus() → HTTP 查询 Prometheus
  ▼
metrics_head.py:
  GET /api/grafana_health       → Grafana 健康检查
  GET /api/prometheus_health    → Prometheus 健康检查
  │
  │ Dashboard 的 Metrics 页面可跳转到 Grafana（外部链接）
  ▼
```

---

### 3.13 Runtime Env 数据流

```
GCS RuntimeEnvTable
  │
  │ StateDataSourceClient
  ▼
state_head.py: GET /api/v0/runtime_envs        [~L160]
  │ Runtime Env 信息列表
```

---

## 四、Dashboard 模块清单

| 模块 | head.py 文件 | 主要职责 |
|------|-------------|----------|
| node | `modules/node/node_head.py` | **核心数据采集**：GCS PubSub订阅（Actor/Node/ResourceUsage）、Raylet gRPC（NodeStats）、DataOrganizer 组织 |
| state | `modules/state/state_head.py` | **State API（/api/v0/*）**：Task/Worker/Object/Log/PlacementGroup/RuntimeEnv/ClusterEvent 等查询 |
| job | `modules/job/job_head.py` | **Job 管理**：提交/停止/查询 Job |
| event | `modules/event/event_head.py` | **事件收集与查询**：Cluster Event + Pipeline Event |
| serve | `modules/serve/serve_head.py` | **Serve 管理**：应用部署/删除/更新/扩缩容 |
| train | `modules/train/train_head.py` | **Train 查询**：V1/V2 Train Run 信息 |
| data | `modules/data/data_head.py` | **Ray Data**：Dataset 信息查询 |
| metrics | `modules/metrics/metrics_head.py` | **监控集成**：Prometheus/Grafana 健康检查 |
| reporter | `modules/reporter/reporter_head.py` | **Reporter Agent**：物理资源数据采集上报 |
| aggregator | `modules/aggregator/aggregator_head.py` | **数据聚合**（旧版） |
| usage_stats | `modules/usage_stats/usage_stats_head.py` | **用量统计** |

---

## 五、node_head.py 数据采集方法一览

| 方法 | 数据类型 | 数据源机制 | 写入 DataSource |
|------|----------|-----------|-----------------|
| `_update_nodes()` | 集群节点 | `GcsAioNodeInfoSubscriber` + `async_get_all_node_info()` | `DataSource.nodes`, `DataSource.node_stats` |
| `_update_actors()` | Actor | `GcsAioActorSubscriber` + `_get_all_actors()` | `DataSource.actors`, `DataSource.node_actors` |
| `_update_node_stats()` | Raylet 统计 | gRPC `GetNodeStats` → 定期遍历 ALIVE 节点 | `DataSource.node_stats` |
| `_update_node_physical_stats()` | 物理资源 | `GcsAioResourceUsageSubscriber` → Reporter Agent 上报 | `DataSource.node_physical_stats` |

每个方法都遵循相同的模式：
1. 先 subscribe（建立 PubSub 通道）
2. 再 get-all（全量拉取，防止 TOCTOU）
3. 进入无限循环 poll 增量更新
4. 写入 DataSource 静态字典

---

## 六、state_head.py API 端点一览

| 端点                              | 说明                 |
| ------------------------------- | ------------------ |
| `GET /api/v0/actors`            | Actor 列表           |
| `GET /api/v0/actors/summarize`  | Actor 统计摘要         |
| `GET /api/v0/jobs`              | Job 列表             |
| `GET /api/v0/nodes`             | Node 列表            |
| `GET /api/v0/workers`           | Worker 列表          |
| `GET /api/v0/tasks`             | Task 列表            |
| `GET /api/v0/tasks/summarize`   | Task 统计摘要          |
| `GET /api/v0/tasks/timeline`    | Task 时间线           |
| `GET /api/v0/objects`           | Object 列表          |
| `GET /api/v0/objects/summarize` | Object 统计摘要        |
| `GET /api/v0/placement_groups`  | Placement Group 列表 |
| `GET /api/v0/runtime_envs`      | Runtime Env 列表     |
| `GET /api/v0/logs`              | 日志文件列表             |
| `GET /api/v0/logs/{media_type}` | 流式日志内容             |
| `GET /api/v0/cluster_events`    | 集群事件               |
| `GET /api/v0/delay/{delay_s}`   | 测试延迟响应             |

> State API 支持通用过滤参数：`filter_keys`, `filter_predicates`, `filter_values`, `limit`, `detail`

---

## 七、DataOrganizer 聚合逻辑详解

### 7.1 organize() — Worker 数据合并

```
输入: DataSource.node_physical_stats + DataSource.node_stats
处理: _extract_workers_for_node()
  1. 从 node_stats.coreWorkersStats 提取 PID → core_worker_stats 映射
  2. 从 node_physical_stats.workers 提取 PID → physical worker 映射
  3. 按 PID 关联，将 core_worker_stats 合入 physical worker
  4. 添加 language (来自 core_worker) 和 jobId (来自 core_worker)
输出: DataSource.node_workers + DataSource.core_worker_stats
```

### 7.2 _get_actor_info() — Actor 数据聚合

```
输入: DataSource.actors[actor_id] (基础 Actor 信息)
聚合:
  1. DataSource.core_worker_stats[worker_id] → 进程级别信息 (pid, language)
  2. DataSource.node_physical_stats[node_id].workers → 按 PID 匹配 processStats
  3. DataSource.node_physical_stats[node_id].gpus → 按 PID 匹配 GPU 利用率
  4. DataSource.node_physical_stats[node_id].mem → 内存信息
  5. parse_pg_formatted_resources_to_original(requiredResources) → 资源格式转换
输出: 聚合后的 Actor 详情 dict
```

### 7.3 get_node_info() — Node 数据聚合

```
输入: DataSource.node_physical_stats + DataSource.node_stats + DataSource.nodes
聚合:
  1. node_physical_stats 作为基础
  2. node_stats 合入 node_physical_stats["raylet"] 字段
  3. objectStoreBytesUsed/Avail → raylet.object_store_used/available_memory
  4. GcsNodeInfo 合入 raylet 字段 (state, deathInfo, stateMessage)
  5. 若非 summary:
     - DataSource.node_actors[node_id] → 合入 actors (每个 actor 再调 _get_actor_info)
     - DataSource.node_workers[node_id] → 合入 workers
输出: 节点详情或摘要 dict
```

### 7.4 purge() — 过期数据清理

```
定期执行，只保留 DataSource.nodes 中 state == "ALIVE" 的节点数据：
  - 清除 node_stats 中非 ALIVE 节点的数据
  - 清除 node_physical_stats 中非 ALIVE 节点的数据
```

---

## 八、两套 API 对比

### 何时用 Legacy API（node_head.py）

| 端点 | 前端使用场景 |
|------|------------|
| `GET /nodes` | Cluster → Nodes 列表页 |
| `GET /nodes/{id}` | Node 详情页 |
| `GET /logical/actors` | Actors 列表页 |
| `GET /logical/actors/{id}` | **Actor 详情页**（主要用户入口） |

### 何时用 State API（state_head.py）

| 端点 | 前端使用场景 |
|------|------------|
| `GET /api/v0/tasks` | Task 列表/详情（嵌入 Actor/Job 页面） |
| `GET /api/v0/workers` | Worker 查询 |
| `GET /api/v0/logs` | 日志列表/查看器 |
| `GET /api/v0/objects` | Object 查询 |
| `GET /api/v0/placement_groups` | Placement Group 页面 |
| `GET /api/v0/actors` | Actor 数据的 State API 版（部分场景使用） |
| `GET /api/v0/tasks/summarize` | Job 详情页中的 Task 统计 |
| `GET /api/v0/cluster_events` | 集群事件 |

### 为什么两套并存？

- **Legacy API** (`node_head.py`)：DashboardHead 持续订阅 GCS → 数据实时缓存在内存 → 响应快但数据可能被 purge 清理
- **State API** (`state_head.py`)：每次请求直接查询 GCS → 数据最新但响应可能较慢 → 支持更丰富的过滤/分页/聚合

---

## 九、前端页面路由总览

来自 `App.tsx` 的路由定义：

| URL 模式 | 页面组件 | 数据来源 |
|----------|---------|----------|
| `/cluster` | `Nodes` | `/nodes?view=summary` |
| `/cluster/nodes/:id` | `NodeDetailPage` | `/nodes/${id}` |
| `/actors` | `Actors` | `/logical/actors` |
| `/actors/:actorId` | `ActorDetailPage` | `/logical/actors/${actorId}` |
| `/actors/:actorId/tasks/:taskId` | `TaskPage` | `/api/v0/tasks` |
| `/jobs` | `JobList` | `/api/jobs/` |
| `/jobs/:id` | `JobDetailLayout` | `/api/jobs/${id}` |
| `/jobs/:id/tasks/:taskId` | `TaskPage` | `/api/v0/tasks` |
| `/tasks/:taskId` | `TaskPage` | `/api/v0/tasks` |
| `/logs` | `StateApiLogsListPage` | `/api/v0/logs` |
| `/logs/viewer` | `StateApiLogViewerPage` | `/api/v0/logs/{path}` |
| `/serve` | `ServeSideTabLayout` | `/api/serve/applications/` |
| `/metrics` | `Metrics` | `/api/grafana_health`, `/api/prometheus_health` |

---

## 十、关键源码文件索引

### 后端

| 文件 | 职责 |
|------|------|
| [modules/node/node_head.py](../python/ray/dashboard/modules/node/node_head.py) | DashboardHead：GCS PubSub 订阅、gRPC 数据采集、Legacy API 端点 |
| [modules/node/datacenter.py](../python/ray/dashboard/modules/node/datacenter.py) | DataSource + DataOrganizer：数据存储与聚合 |
| [modules/state/state_head.py](../python/ray/dashboard/modules/state/state_head.py) | State API：`/api/v0/*` 端点 |
| [modules/state/state_aggregator.py](../python/ray/dashboard/state_aggregator.py) | StateAPIManager + StateDataSourceClient |
| [modules/event/event_head.py](../python/ray/dashboard/modules/event/event_head.py) | 事件收集与 Cluster Event API |
| [modules/job/job_head.py](../python/ray/dashboard/modules/job/job_head.py) | Job 管理 API |
| [modules/serve/serve_head.py](../python/ray/dashboard/modules/serve/serve_head.py) | Serve 管理 API |

### 前端

| 文件 | 职责 |
|------|------|
| [client/src/App.tsx](../python/ray/dashboard/client/src/App.tsx) | 路由定义 |
| [client/src/pages/actor/](../python/ray/dashboard/client/src/pages/actor/) | Actor 页面组件 |
| [client/src/pages/node/](../python/ray/dashboard/client/src/pages/node/) | Node 页面组件 |
| [client/src/pages/state/task.tsx](../python/ray/dashboard/client/src/pages/state/task.tsx) | Task 页面组件 |
| [client/src/pages/log/](../python/ray/dashboard/client/src/pages/log/) | Log 页面组件 |
| [client/src/service/actor.ts](../python/ray/dashboard/client/src/service/actor.ts) | Actor API service |
| [client/src/service/node.ts](../python/ray/dashboard/client/src/service/node.ts) | Node API service |
| [client/src/service/task.ts](../python/ray/dashboard/client/src/service/task.ts) | Task API service |
| [client/src/service/log.ts](../python/ray/dashboard/client/src/service/log.ts) | Log API service |
| [client/src/service/job.ts](../python/ray/dashboard/client/src/service/job.ts) | Job API service |
| [client/src/service/serve.ts](../python/ray/dashboard/client/src/service/serve.ts) | Serve API service |

---

## 十一、数据流走向图（精简版）

```
                          ┌─────────┐
                          │   GCS   │
                          │(各种Table│
                          │ + PubSub│
                          │ 通道)   │
                          └────┬────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ActorSubscriber     NodeInfoSubscriber    ResourceUsageSubscriber
          │                    │                    │
          ▼                    ▼                    ▼
   _update_actors()    _update_nodes()     _update_node_physical_stats()
          │                    │                    │
          ▼                    ▼                    ▼
  DataSource.actors    DataSource.nodes     DataSource.node_physical_stats
          │                    │                    │
          │                    │                    │
          │         ┌──────────┴──────────┐         │
          │         │  _update_node_stats │         │
          │         │  (gRPC → Raylet)    │         │
          │         └──────────┬──────────┘         │
          │                    │                    │
          │                    ▼                    │
          │         DataSource.node_stats           │
          │                    │                    │
          │    ┌───────────────┼────────────────┐   │
          │    │               │                │   │
          │    ▼               ▼                ▼   │
          │ organize()   get_node_info()   get_actor_infos()
          │    │               │                │   │
          │    ▼               ▼                ▼   │
          │ node_workers   Node API         Actor API
          │ + core_worker  /nodes           /logical/actors
          │   _stats       /nodes/{id}      /logical/actors/{id}
          │    │                                │
          │    │                                │
          └────────────── DataOrganizer ─────────┘
                               │
                               │  同时...
                               │
                          ┌────┴────┐
                          │ State   │
                          │ API     │
                          │Manager  │
                          └────┬────┘
                               │ 直接查询 GCS
                               ▼
                    /api/v0/tasks
                    /api/v0/workers
                    /api/v0/logs
                    /api/v0/objects
                    /api/v0/placement_groups
                    /api/v0/cluster_events
                               │
                               ▼
                         Frontend (SWR)
```