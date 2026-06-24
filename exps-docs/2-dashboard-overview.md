# Ray Dashboard 进程架构与启动链路全解析

> 基于 Ray 2.53.0 源码，从 `ray start --head` 到每一个进程跑起来的完整链路，以及进程/线程职责与通信关系的全景图。

---

## 一、进程架构全景

```
┌───────────────────────────────────────────────────────────────────┐
│  Head Node                                                        │
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌────────────────────────┐  │
│  │ Raylet (C++)│   │  GCS (C++)  │   │ Dashboard Head 主进程  │  │
│  │             │   │             │   │  (aiohttp + 反向代理)   │  │
│  └─────────────┘   └─────────────┘   └────────────┬───────────┘  │
│           │                                      │               │
│           │ spawn via             ┌───────────────┼────────────┐  │
│           │ AgentManager          │ Subprocess Module 进程群   │  │
│           │                      │ (每个 spawn 独立进程)       │  │
│           │                      │ ┌──────────┐ ┌───────────┐ │  │
│           │                      │ │ NodeHead │ │  JobHead  │ │  │
│  ┌────────▼──────────┐           │ ├──────────┤ ├───────────┤ │  │
│  │ Dashboard Agent   │           │ │StateHead │ │ ServeHead │ │  │
│  │ (每节点 1 个)     │           │ ├──────────┤ ├───────────┤ │  │
│  │ GRPC + HTTP server│           │ │ReportHead│ │MetricsHead│ │  │
│  └───────────────────┘           │ ├──────────┤ ├───────────┤ │  │
│           │                      │ │ DataHead │ │ TrainHead │ │  │
│           │                      │ └──────────┘ └───────────┘ │  │
│           │                      └───────────────┴────────────┘  │
│           │                                                      │
│  ┌────────▼──────────┐                                          │
│  │ Runtime Env Agent │  ← 也由 AgentManager spawn               │
│  └───────────────────┘                                          │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────┐
│  Worker Node                      │
│                                   │
│  ┌─────────────┐  ┌────────────┐  │
│  │ Raylet (C++)│  │Dashboard   │  │
│  │             │  │Agent       │  │
│  └─────────────┘  └────────────┘  │
│                                   │
│  ┌─────────────┐                  │
│  │Runtime Env  │                  │
│  │Agent        │                  │
│  └─────────────┘                  │
└───────────────────────────────────┘
```

### 进程数量汇总

| 节点类型            | 进程名                  | 数量     | 线程数                                |
| --------------- | -------------------- | ------ | ---------------------------------- |
| **Head Node**   | Dashboard Head 主进程   | 1      | ~2 (async 主线程 + TPE 1 worker)      |
|                 | Subprocess Module 进程 | 8      | ~2 each (async 主线程 + TPE 1 worker) |
|                 | Dashboard Agent      | 1      | ~2 (async 主线程 + TPE 1 worker)      |
|                 | Runtime Env Agent    | 1      | ~2                                 |
|                 | **合计**               | **11** | **~22**                            |
| **Worker Node** | Dashboard Agent      | 1      | ~2                                 |
|                 | Runtime Env Agent    | 1      | ~2                                 |
|                 | **合计**               | **2**  | **~4**                             |

> 以上不含 Raylet、GCS（C++ 进程），也不含 Plasma Store。

---

## 二、整体启动时序图

```
用户执行 ray start --head (或 ray.init())
    │
    ▼
Node.start_head_node_processes_if_needed  ─── (node.py:1360)
    │
    ├──① start_gcs_server()                    ──→ GCS (C++ 进程) 启动
    │
    ├──② start_api_server()                    ──→ Dashboard Head (Python) 启动
    │     │
    │     ▼
    │   services.start_api_server()             ─── (services.py:1143)
    │     │  构造命令行:
    │     │    python -u dashboard.py
    │     │      --host=... --port=8265 --port-retries=50
    │     │      --gcs-address=... --cluster-id-hex=... --node-ip-address=...
    │     │      [--minimal]  (如果依赖缺失)
    │     │
    │     │  → start_ray_process(command) → fork 子进程
    │     │
    │     │  ── 轮询 GCS KV 等待 Dashboard 就绪 ──
    │     │  while timeout 未到:
    │     │    url = internal_kv_get("dashboard", ns="dashboard")
    │     │    if url != None → break  ← Dashboard 已注册 URL
    │     │    if process 已退出 → break  ← Dashboard 启动失败
    │     │    sleep(0.1)
    │     │
    │     ▼
    │   Dashboard Head 子进程                   ─── (dashboard.py:__main__)
    │     │
    │     ▼
    │   DashboardHead.run()                     ─── (head.py:390)
    │     │
    │     ├──③ _load_modules()
    │     │     ├── DashboardHeadModule: UsageStatsHead 等 (主进程内)
    │     │     └── SubprocessModuleHandle ×8    (将 spawn 子进程)
    │     │
    │     ├──④ 并行 spawn 所有 SubprocessModule       (head.py:401)
    │     │     │
    │     │     │  SubprocessModuleHandle.start_module()
    │     │     │    │  multiprocessing.Pipe() → 创建父子通信管道
    │     │     │    │  multiprocessing.Process(spawn, target=run_module)
    │     │     │    │  → fork+exec 子进程
    │     │     │    │
    │     │     │    └── NodeHead 进程    (Unix Socket aiohttp server)
    │     │    ├── StateHead 进程    (Unix Socket aiohttp server)
    │     │    ├── ReportHead 进程   (Unix Socket aiohttp server)
    │     │    ├── JobHead 进程      (Unix Socket aiohttp server)
    │     │    ├── ServeHead 进程    (Unix Socket aiohttp server)
    │     │    ├── MetricsHead 进程  (Unix Socket aiohttp server)
    │     │    ├── DataHead 进程     (Unix Socket aiohttp server)
    │     │    └── TrainHead 进程    (Unix Socket aiohttp server)
    │     │
    │     ├──⑤ 串行 wait_for_module_ready()            (head.py:404)
    │     │     │  每个子进程 run_module_inner():
    │     │     │    module = cls(config)
    │     │     │    await module.run()  ← 启动 aiohttp Unix Socket server
    │     │     │    child_conn.send(None)  ← 通过 Pipe 发 ready 信号
    │     │     │    child_conn.close()
    │     │     │
    │     │     │  父进程:
    │     │     │    parent_conn.recv()  ← 收到 None = ready
    │     │     │    parent_conn.close()
    │     │     │    http_session = aiohttp.ClientSession() → Unix Socket
    │     │     │    health_check_task = loop.create_task(periodic_check)
    │     │     │
    │     │     └── 全部子进程就绪 ✓
    │     │
    │     ├──⑥ _configure_http_server()                (head.py:436)
    │     │     │
    │     │     ▼ HttpServerDashboardHead.run()
    │     │     │  (http_server_head.py:356)
    │     │     │
    │     │     │  1. DashboardHeadRouteTable.bind(m)    → 主进程内路由
    │     │     │  2. SubprocessRouteTable.bind(handle)  → 代理路由
    │     │     │  3. aiohttp.web.Application(middlewares=[auth, metrics, path_clean])
    │     │     │  4. app.add_routes(head_routes + subprocess_proxy_routes)
    │     │     │  5. aiohttp.web.TCPSite → 监听 TCP 端口 (默认 8265)
    │     │     │
    │     │     └── HTTP 服务器就绪 ✓
    │     │
    │     ├──⑦ 注册 Dashboard URL 到 GCS                (head.py:459)
    │     │     │  gcs_client.internal_kv_put(
    │     │     │    key="dashboard", ns="dashboard",
    │     │     │    value="http://10.0.0.1:8265"
    │     │     │  )
    │     │     │
    │     │     └── start_api_server() 的轮询读到 URL → 返回给 Node ✓
    │     │
    │     └──⑧ asyncio.gather(
    │            _gcs_check_alive(),          ← 每 5s 检查 GCS 存活
    │            dashboard_head_modules.run(), ← UsageStatsHead 等
    │            ...                           ← 永久阻塞，主事件循环
    │          )
    │
    ├──⑨ start_raylet()                        ──→ Raylet (C++ 进程) 启动
    │     │
    │     ▼ services.start_raylet()             ─── (services.py:1518)
    │     │
    │     │  构造 Raylet 命令行:
    │     │    raylet --raylet_socket_name=...
    │     │      --dashboard_agent_command="<整条agent命令作为字符串>"
    │     │      --runtime_env_agent_command="<整条runtime_env命令>"
    │     │
    │     │  dashboard_agent_command 的构造 (services.py:1761):
    │     │    python -u dashboard/agent.py
    │     │      --node-ip-address=... --grpc-port=...
    │     │      --listen-port=... --gcs-address=...
    │     │      --node-manager-port=RAY_NODE_MANAGER_PORT_PLACEHOLDER ← 占位符!
    │     │      [--minimal]
    │     │
    │     │  嵌入 Raylet 参数 (services.py:1908):
    │     │    --dashboard_agent_command="python -u ... --node-manager-port=PLACEHOLDER ..."
    │     │
    │     ▼
    │   Raylet 进程启动                          ─── (raylet main.cc)
    │     │
    │     ├── NodeManager 构造函数                ─── (node_manager.cc:269)
    │     │     │
    │     │     ├──⑩ CreateDashboardAgentManager()     (node_manager.cc:3271)
    │     │     │     │
    │     │     │     │  1. 解析 dashboard_agent_command
    │     │     │     │  2. 查找 RAY_NODE_MANAGER_PORT_PLACEHOLDER
    │     │     │     │  3. 替换为 Raylet 实际绑定的端口 (GetServerPort())
    │     │     │     │  4. Options{node_id, "dashboard_agent", command, fate_shares=true}
    │     │     │     │
    │     │     │     ▼ AgentManager.StartAgent()      (agent_manager.cc:31)
    │     │     │     │
    │     │     │     │  C++ Process fork+exec:
    │     │     │     │    argv = [python, -u, agent.py, --node-ip-address=..., ...]
    │     │     │     │    env = {RAY_NODE_ID, RAY_RAYLET_PID,
    │     │     │     │           RAY_enable_pipe_based_agent_to_parent_health_check}
    │     │     │     │    pipe_to_stdin = true  ← stdin pipe 连到 raylet
    │     │     │     │
    │     │     │     │  monitor_thread_ (C++ std::thread):
    │     │     │     │    → process_.Wait() 等待 agent 退出
    │     │     │     │    → agent 死 + fate_shares=true → shutdown_raylet + 10s force kill
    │     │     │     │
    │     │     │     └── Dashboard Agent 子进程已启动 ✓
    │     │     │
    │     │     ├──⑪ CreateRuntimeEnvAgentManager()  ← 类似流程，spawn Runtime Env Agent
    │     │     │     └── fate_shares=true
    │     │     │
    │     │     └── Raylet 主逻辑继续运行...
    │     │
    │     ▼
    │   Dashboard Agent 子进程启动               ─── (agent.py:__main__)
    │     │
    │     ▼ DashboardAgent.__init__()
    │     │  │  GcsClient(address=gcs_address) → 连接 GCS
    │     │  │  获取 node_info → 判断 is_head
    │     │  │  _init_non_minimal():
    │     │  │    → aiogrpc.server() (GRPC server)
    │     │  │    → HttpServerAgent(ip, listen_port) (HTTP server)
    │     │  │
    │     │  ▼ DashboardAgent.run()             ─── (agent.py:188)
    │     │     │
    │     │     ├──⑫ await server.start()       ← aiogrpc GRPC server 启动
    │     │     │
    │     │     ├──⑬ await http_server.start(modules) ← aiohttp HTTP server 启动
    │     │     │     │  aiohttp TCPSite 监听 listen_port (默认 52365)
    │     │     │     │  注册 Agent 模块路由 (HealthzAgent, JobAgent, LogAgent...)
    │     │     │     │
    │     │     │     └── Agent HTTP server 就绪 ✓
    │     │     │
    │     │     ├──⑭ _load_modules()            ← 加载 DashboardAgentModule
    │     │     │     ├── HealthzAgent
    │     │     │     ├── JobAgent
    │     │     │     ├── LogAgent / LogAgentV1Grpc
    │     │     │     ├── EventAgent
    │     │     │     └── TestAgent (测试用)
    │     │     │
    │     │     ├──⑮ 注册 Agent 地址到 GCS KV           (agent.py:218)
    │     │     │     │  gcs_client.async_internal_kv_put(
    │     │     │     │    "DASHBOARD_AGENT_ADDR_NODE_ID_PREFIX:<node_id>",
    │     │     │     │    json.dumps([ip, http_port, grpc_port]),
    │     │     │     │    ns="dashboard"
    │     │     │     │  )
    │     │     │     │  gcs_client.async_internal_kv_put(
    │     │     │     │    "DASHBOARD_AGENT_ADDR_IP_PREFIX:<ip>",
    │     │     │     │    json.dumps([node_id, http_port, grpc_port]),
    │     │     │     │    ns="dashboard"
    │     │     │     │  )
    │     │     │     │
    │     │     │     └── NodeHead 可从 GCS KV 读到 Agent 地址 ✓
    │     │     │
    │     │     └──⑯ asyncio.gather(
    │     │           modules.run(server),              ← Agent 模块运行
    │     │           create_check_raylet_task(),       ← 监测 raylet 存活
    │     │           server.wait_for_termination(),    ← GRPC 永久阻塞
    │     │         )
    │     │
    │     └── Dashboard Agent 就绪 ✓
    │
    └── Node._webui_url = "http://10.0.0.1:8265" ← 用户可访问 Dashboard ✓
```

---

## 三、进程详解

### 3.1 Dashboard Head 主进程

| 项目 | 详情 |
|------|------|
| **入口文件** | `dashboard.py:__main__` → `Dashboard.run()` → `DashboardHead.run()` |
| **数量** | 每集群 1 个，只运行在 Head Node |
| **启动方式** | `services.start_api_server()` 构造命令行 → `start_ray_process()` → fork 子进程 |
| **核心职责** | 1. 运行 aiohttp HTTP 服务器 (TCP 端口 8265)，对外提供 Web UI + REST API<br>2. 作为 HTTP 反向代理，将子进程路由的请求转发到 Unix Socket<br>3. 加载 DashboardHeadModule（UsageStatsHead 等）在主进程内运行<br>4. 管理 SubprocessModuleHandle（spawn、健康检查、重启）<br>5. 定期检查 GCS 存活 (`_gcs_check_alive` 每 5s)<br>6. 注册 Dashboard URL 到 GCS KV |
| **线程模型** | 主线程 = asyncio 事件循环（单线程异步）<br>+ ThreadPoolExecutor(max_workers=1, `RAY_DASHBOARD_DASHBOARD_HEAD_TPE_MAX_WORKERS`) |

**线程模型图**：

```
┌────────────────────────────────────────────────────────┐
│  Dashboard Head Process                                │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Main Thread (asyncio Event Loop)                 │ │
│  │  ├── aiohttp.web TCPSite (8265)                  │ │
│  │  │     ├── / → index.html (静态文件)              │ │
│  │  │     ├── /api/* → DashboardHeadModule 路由      │ │
│  │  │     └── /api/* → SubprocessModule 代理路由     │ │
│  │  │         → proxy_http() / proxy_stream()       │ │
│  │  │         → Unix Socket HTTP 转发到子进程        │ │
│  │  ├── _gcs_check_alive() @async_loop_forever(5s) │ │
│  │  ├── _record_dashboard_metrics()                 │ │
│  │  ├── health_check_tasks ×8 (每子进程每1s)         │ │
│  │  ├── DashboardHeadModule.run()                   │ │
│  │  └── run_in_executor() ────────────┐             │ │
│  └────────────────────────────────────┼─────────────┘ │
│                                       │               │
│  ┌────────────────────────────────────▼─────────────┐ │
│  │ ThreadPoolExecutor (max_workers=1)               │ │
│  │  └── CPU 密集操作: protobuf 解析, JSON 处理等   │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

> **关键设计**: TPE 默认只有 1 个 worker thread。注释明确写了 "intentionally constrained to just 1 thread...to limit its concurrency, therefore reducing potential for GIL contention"。 (head.py:41-46)

### 3.2 Subprocess Module 进程（×8）

每个都是**独立进程**，通过 `multiprocessing.Process(spawn)` 创建，运行独立的 aiohttp 服务器，监听 **Unix Socket** (Linux/macOS) 或 **Named Pipe** (Windows)。

| Module | 类名 | 文件 | 职责 |
|--------|------|------|------|
| Node Info | `NodeHead` | `modules/node/node_head.py` | 集群节点/Actor 信息聚合；订阅 GCS 的 Node/Actor/Resource 变更 |
| State API | `StateHead` | `modules/state/state_head.py` | 提供 State API (actors/jobs/tasks/workers observability 查询) |
| Reporter | `ReportHead` | `modules/reporter/reporter_head.py` | 节点资源使用统计收集与上报 |
| Job | `JobHead` | `modules/job/job_head.py` | Job 提交与管理，启动 Job Supervisor |
| Serve | `ServeHead` | `modules/serve/serve_head.py` | Ray Serve 部署管理 |
| Metrics | `MetricsHead` | `modules/metrics/metrics_head.py` | Prometheus/Grafana 集成，指标导出 |
| Data | `DataHead` | `modules/data/data_head.py` | Dataset 相关操作 |
| Train | `TrainHead` | `modules/train/train_head.py` | Ray Train 工作负载管理 |

**每个子进程的线程模型**：

```
┌────────────────────────────────────────────────────────┐
│  SubprocessModule Process (e.g. NodeHead)              │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Main Thread (asyncio Event Loop)                 │ │
│  │  ├── aiohttp.web.UnixSite (socket file)          │ │
│  │  │     ├── /api/healthz → health check           │ │
│  │  │     └── /api/node/* → NodeHead 业务路由       │ │
│  │  ├── GcsAioNodeInfoSubscriber.poll()             │ │
│  │  ├── GcsAioActorSubscriber.poll()                │ │
│  │  ├── GcsAioResourceUsageSubscriber.poll()        │ │
│  │  ├── @async_loop_forever background tasks        │ │
│  │  ├── _detect_parent_process_death() (每1s)       │ │
│  │  └── run_in_executor() ────────────┐             │ │
│  └────────────────────────────────────┼─────────────┘ │
│                                       │               │
│  ┌────────────────────────────────────▼─────────────┐ │
│  │ ThreadPoolExecutor (max_workers=1)               │ │
│  │  ├── NodeHead: _node_executor (env: TPE_MAX=1)  │ │
│  │  ├── NodeHead: _actor_executor (固定 max=1)     │ │
│  │  └── 其他模块: 各自有 1 worker TPE              │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

**spawn 方式的关键细节** (handle.py:86-87):

> "Force using spawn because Ray C bindings have static variables that need to be re-initialized for a new process." — fork 会继承旧的全局状态，spawn 会重新初始化。

**子进程入口** (module.py:231 → 201):

```
run_module(cls, config, incarnation, child_conn):
  1. setproctitle("ray-dashboard-{ModuleName}-{incarnation}")
  2. setup_component_logger(...) → 独立日志文件 dashboard_{Module}.log
  3. run_module_inner():
     a. module = cls(config)                     ← 创建模块实例
     b. _detect_parent_process_death_task        ← 监听父进程存活
     c. await module.run()                       ← 启动 aiohttp Unix Socket server
     d. child_conn.send(None)                    ← 发 ready 信号给父进程
     e. child_conn.close()
```

### 3.3 DashboardHeadModule（主进程内运行）

只有 2 个模块在主进程内直接运行（不 spawn 子进程）：

| Module | 类名 | 文件 | 职责 |
|--------|------|------|------|
| Usage Stats | `UsageStatsHead` | `modules/usage_stats/usage_stats_head.py` | 使用统计记录 |
| Test | `TestHead` | `modules/tests/test_head.py` | 测试模块 (非生产) |

### 3.4 Dashboard Agent 进程

| 项目 | 详情 |
|------|------|
| **入口文件** | `agent.py:__main__` → `DashboardAgent.__init__()` → `DashboardAgent.run()` |
| **数量** | 每 Ray 节点 1 个 (Head + Worker 都有) |
| **启动方式** | 由 Raylet 的 `AgentManager` (C++) spawn 启动 |
| **启动命令构造** | Python `services.py:1761` 构造 → 嵌入 Raylet `--dashboard_agent_command` 参数 → Raylet C++ 层替换 `PLACEHOLDER` → `AgentManager.StartAgent()` fork+exec |
| **核心职责** | 1. 收集本节点 stats/metrics<br>2. 提供 HTTP API 给 Dashboard Head 调用<br>3. GRPC server 提供节点级服务 (非 minimal)<br>4. Agent 地址注册到 GCS Internal KV<br>5. 监测 Raylet 存活 |
| **Agent 模块** | `HealthzAgent`, `JobAgent`, `LogAgent`, `LogAgentV1Grpc`, `EventAgent` |
| **线程模型** | 主线程 = asyncio 事件循环 + HTTP server (aiohttp TCPSite)<br>+ GRPC server (asyncio-based)<br>+ TPE worker |

**线程模型图**：

```
┌────────────────────────────────────────────────────────┐
│  Dashboard Agent Process                               │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Main Thread (asyncio Event Loop)                 │ │
│  │  ├── aiohttp.web TCPSite (listen_port=52365)     │ │
│  │  │     ├── /api/healthz → HealthzAgent           │ │
│  │  │     ├── /api/logs/* → LogAgent                │ │
│  │  │     └── /api/job/* → JobAgent                 │ │
│  │  ├── aiogrpc.server (grpc_port)                  │ │
│  │  │     ├── LogAgentV1Grpc service                │ │
│  │  │     └── 其他 GRPC services                    │ │
│  │  ├── create_check_raylet_task()                  │ │
│  │  │     → 监测 Raylet 存活 (log_dir socket 文件) │ │
│  │  ├── modules.run(server)                         │ │
│  │  └── run_in_executor() ────────────┐             │ │
│  └────────────────────────────────────┼─────────────┘ │
│                                       │               │
│  ┌────────────────────────────────────▼─────────────┐ │
│  │ ThreadPoolExecutor                               │ │
│  │  └── CPU 密集操作                                │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

> **Fate-sharing**: Agent 和 Raylet 是命运共享的 — Agent 崩溃 → Raylet 也退出 (fate_shares=true)。 (node_manager.cc:3294-3298)

---

## 四、进程间通信关系

```
                    ┌──────────────────────────────────────────┐
                    │              GCS (C++ 进程)              │
                    │   PubSub + Internal KV + RPC             │
                    └──────────┬──────────────┬────────────────┘
                               │              │
            ┌──────────────────┤              ├─────────────────┐
            │  GcsAio*Subscriber│             │ GcsClient RPC   │
            │  (async pubsub)  │             │ (async/grpc)    │
            ▼                  ▼              ▼                 ▼
  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
  │ Subprocess Module│  │Dashboard Head│  │Dashboard Agent   │
  │ (NodeHead 等)    │  │  (主进程)     │  │ (每节点)          │
  └──────────────────┘  └──────┬───────┘  └──────────┬───────┘
                               │                      │
          ┌────────────────────┤                      │
          │ Unix Socket / Pipe │   HTTP (TCP)         │
          │ (IPC 反向代理)     │                      │
          ▼                    ▼                      │
  ┌──────────────────┐  ┌──────────────────┐         │
  │ Subprocess Module│  │ Web UI / REST API│         │
  │ (aiohttp server) │  │ (aiohttp TCP)    │◄────────┘
  └──────────────────┘  └──────────────────┘   Agent HTTP API
```

### 通信方式详表

| 通信路径 | 方式 | 说明 |
|----------|------|------|
| Dashboard Head → Subprocess Module | **Unix Socket / Named Pipe** | 主进程作为 HTTP 反向代理，前端请求通过 Unix Socket 转发到子进程 aiohttp server |
| Dashboard Head → GCS | **GcsClient (async grpc)** | 查询集群信息、检查存活、注册 URL |
| Subprocess Module → GCS | **GcsAio*Subscriber + GcsClient** | `GcsAioNodeInfoSubscriber`, `GcsAioActorSubscriber`, `GcsAioResourceUsageSubscriber` — 纯 async 实现，非线程 |
| Dashboard Agent → GCS | **GcsClient + Internal KV** | 注册地址 (`DASHBOARD_AGENT_ADDR_*`)、上报节点信息 |
| Dashboard Head → Dashboard Agent | **HTTP (TCP)** | 从 GCS KV 读 Agent 地址 → HTTP 调用 Agent API 获取节点级数据 |
| Raylet → Dashboard Agent | **spawn + stdin pipe** | C++ AgentManager 管理 Agent 进程生命周期；stdin pipe 用于 Agent 检测 Raylet 死亡 |
| Dashboard Head → SubprocessModule (初始化) | **multiprocessing.Pipe** | 子进程 `child_conn.send(None)` 发 ready 信号；之后不再使用 Pipe |
| Dashboard Head → SubprocessModule (运行时) | **HTTP healthz (Unix Socket)** | 每 1s 请求 `/api/healthz`，检测子进程存活与响应延迟 |

---

## 五、进程间"握手"信号汇总

| 启动者 | 被启动者 | 握手/就绪信号 | 通信方式 |
|--------|----------|--------------|----------|
| `Node` (Python) | Dashboard Head | Head 写 `DASHBOARD_ADDRESS` 到 GCS KV → `start_api_server()` 轮询 KV 读 URL | GCS KV (同步轮询, 每 0.1s) |
| Dashboard Head | SubprocessModule | 子进程 `child_conn.send(None)` → 父进程 `parent_conn.recv()` | `multiprocessing.Pipe` (同步) |
| Dashboard Head | SubprocessModule | 周期性 `/api/healthz` → HTTP 200 OK | Unix Socket HTTP (异步, 每 1s) |
| Dashboard Head | SubprocessModule | 不健康 → `destroy_module()` → kill → `start_module()` 重启 | `Process.kill()` + re-spawn |
| Raylet (C++) | Dashboard Agent | 无显式握手；靠 fate-sharing (agent 死 → raylet 死) | C++ `Process.Wait()` |
| Raylet (C++) | Dashboard Agent | stdin pipe (agent 读 stdin EOF 检测 raylet 死亡) | pipe (OS-level) |
| Dashboard Agent | GCS | `async_internal_kv_put(DASHBOARD_AGENT_ADDR_*)` | GCS KV (async grpc) |
| Dashboard Head → Agent | 从 GCS KV 读 Agent 地址 → HTTP 调用 Agent API | GCS KV + HTTP TCP |

---

## 六、健康检查与重启机制

### 6.1 SubprocessModule 健康检查

```
每 1s 循环:
  ┌─ _do_once_health_check()
  │   ├─ 检查 process.exitcode → 非None = 进程已退出 → unhealthy
  │   └─ HTTP GET /api/healthz (Unix Socket) → 非200 → unhealthy
  │
  │ unhealthy:
  │   ├─ log 异常 + 最近 N 行日志文件内容
  │   ├─ destroy_module():
  │   │     ├─ incarnation += 1
  │   │     ├─ process.kill() + process.join()
  │   │     ├─ http_client_session.close()
  │   │     └─ health_check_task.cancel()
  │   ├─ start_module():        ← 重新 spawn 子进程
  │   │     ├─ Pipe() → 新 parent_conn, child_conn
  │   │     └─ Process(spawn, ...) → 新进程启动
  │   └─ wait_for_module_ready(): ← 等 Pipe ready 信号
  │     └─ health_check_task = loop.create_task(periodic_check)
  │
  │ healthy:
  │   └─ sleep(1) → 继续循环
```

> 重启次数无上限 (handle.py:79-82 注释 "max number of restarts: infinite")。

### 6.2 Dashboard Agent 存活监控（双向命运共享）

```
Agent 监控 Raylet:
  ┌─ create_check_raylet_task()            (agent.py:242-244)
  │   └─ 检查方式 1: log_dir 下的 raylet socket 文件存在性
  │   └─ 检查方式 2: gcs_client async check alive
  │   └─ Raylet 死 → agent 自动退出

Raylet 监控 Agent:
  ┌─ monitor_thread_ (C++ std::thread)     (agent_manager.cc:79-111)
  │   └─ process_.Wait() → 等 agent 进程退出
  │   └─ agent 死 + fate_shares=true:
  │       ├─ shutdown_raylet_gracefully_(UNEXPECTED_TERMINATION)
  │       ├─ reason_message = "dashboard_agent failed and raylet fate-shares with it"
  │       └─ delay 10s → QuickExit()  ← 强制自杀
  │
  │   └─ Agent 析构 (~AgentManager):
  │       ├─ fate_shares_ = false  ← 优雅关闭时不触发 fate-sharing
  │       ├─ process_.Kill()
  │       └─ monitor_thread_->join()
```

### 6.3 SubprocessModule 监控父进程

```
_detect_parent_process_death()    (module.py:76-86)
  while True:
    if not self._parent_process.is_alive():  ← 检查父进程 PID
      logger.warning("Parent process died. Exiting...")
      return
    await asyncio.sleep(1)     ← 每 1s 检查一次
```

> Dashboard Head 主进程死 → 所有 SubprocessModule 在 1s 内检测到 → 自行退出。

---

## 七、minimal vs non-minimal 模式

| 特性 | minimal (`pip install ray`) | non-minimal (`pip install ray[default]`) |
|------|---------------------------|------------------------------------------|
| Dashboard Head | 只加载 `UsageStatsHead` (主进程内) | 加载全部 8 个 SubprocessModule + HeadModule |
| SubprocessModule | **不启动** (空列表) | 启动全部 8 个子进程 |
| Dashboard Agent | 无 GRPC server, 无 HTTP server | GRPC server + HTTP server |
| Agent Modules | 只加载 `HealthzAgent` | 加载全部 AgentModule (Job, Log, Event...) |
| Web UI | 不 serve frontend | serve React 前端 |
| Dashboard URL | 仅 `UsageStatsHead.run()` 注册 | 完整 REST API + URL |
| 命令行标记 | `--minimal` | 无标记 |
| 依赖检查 | `dashboard_dependency_error != None` → `--minimal` | 依赖完整 → 无标记 |

---

## 八、日志文件与进程标题

每个进程启动后都有独立的日志和进程标题，便于 `ps` 观察：

| 进程 | 进程标题 (`setproctitle`) | 日志文件 |
|------|-------------------------|---------|
| Dashboard Head | `ray::IDLE` (Python 默认) | `dashboard.log` |
| NodeHead | `ray-dashboard-NodeHead-0` | `dashboard_NodeHead.log` |
| StateHead | `ray-dashboard-StateHead-0` | `dashboard_StateHead.log` |
| ReportHead | `ray-dashboard-ReportHead-0` | `dashboard_ReportHead.log` |
| JobHead | `ray-dashboard-JobHead-0` | `dashboard_JobHead.log` |
| ServeHead | `ray-dashboard-ServeHead-0` | `dashboard_ServeHead.log` |
| MetricsHead | `ray-dashboard-MetricsHead-0` | `dashboard_MetricsHead.log` |
| DataHead | `ray-dashboard-DataHead-0` | `dashboard_DataHead.log` |
| TrainHead | `ray-dashboard-TrainHead-0` | `dashboard_TrainHead.log` |
| Dashboard Agent | `ray::IDLE` (Python 默认) | `dashboard_agent.log` |
| Runtime Env Agent | `ray::IDLE` | `runtime_env_agent.log` |

> 日志轮转: `dashboard_NodeHead.log` → `dashboard_NodeHead.log.1` → `.2` ...
> 受 `--logging-rotate-bytes` 和 `--logging-rotate-backup-count` 控制。

---

## 九、关键源码索引

| 组件 | 文件 | 关键行 |
|------|------|--------|
| Dashboard Head 入口 | `python/ray/dashboard/dashboard.py` | `__main__` → `Dashboard.run()` |
| DashboardHead 主类 | `python/ray/dashboard/head.py` | `DashboardHead.run()` (L390), `_load_modules()` (L173) |
| HTTP 服务器 (Head) | `python/ray/dashboard/http_server_head.py` | `HttpServerDashboardHead.run()` (L356) |
| SubprocessModule 基类 | `python/ray/dashboard/subprocesses/module.py` | `run_module_inner()` (L201), `run()` (L101) |
| SubprocessModuleHandle | `python/ray/dashboard/subprocesses/handle.py` | `start_module()` (L116), `wait_for_module_ready()` (L137) |
| 健康检查 | `python/ray/dashboard/subprocesses/handle.py` | `_do_periodic_health_check()` (L217) |
| NodeHead | `python/ray/dashboard/modules/node/node_head.py` | `_subscribe_for_node_updates()` (L188) |
| Dashboard Agent 入口 | `python/ray/dashboard/agent.py` | `__main__` → `DashboardAgent.run()` (L188) |
| Agent HTTP 服务器 | `python/ray/dashboard/http_server_agent.py` | `HttpServerAgent` (L17) |
| services.start_api_server | `python/ray/_private/services.py` | L1143 |
| services.start_raylet | `python/ray/_private/services.py` | L1518, agent 命令构造 L1761 |
| Node.start_api_server | `python/ray/_private/node.py` | L1081 |
| C++ AgentManager | `src/ray/raylet/agent_manager.h` | `AgentManager` (L49) |
| C++ AgentManager.StartAgent | `src/ray/raylet/agent_manager.cc` | L31 (spawn), L79 (monitor_thread) |
| C++ CreateDashboardAgentManager | `src/ray/raylet/node_manager.cc` | L3271 (替换 PLACEHOLDER, fate_shares=true) |
| GCS PubSub (async) | `python/ray/_private/gcs_pubsub.py` | `GcsAioNodeInfoSubscriber` 等 |