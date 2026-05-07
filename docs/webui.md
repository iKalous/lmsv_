# LMSV_REC WebUI 实现说明（含 Task2 pta_mf 测试）

## 1. 目标与范围

本文档说明 `webui.py` 的实现结构、配置映射、运行控制方式，并记录一次针对 **Task2（模块内组件泛化）+ `COMPARE_MODE=pta_mf`** 的联调测试结果。

适用范围：

- 本地启动 `python webui.py` 的单文件 WebUI。
- 任务配置编辑、保存、启动、停止、日志轮询、结果浏览。
- Task2 的 PTA->MF 分支配置与执行入口。

---

## 2. 总体架构

### 2.1 单文件前后端一体

`webui.py` 同时包含：

- 前端页面模板：`HTML_TEMPLATE`
- 配置 Schema：`FORM_SCHEMA` + `BASE_CONFIG`
- 配置归一化校验：`normalize_config`
- 任务运行管理：`RunManager`
- HTTP 路由处理：`LMSVRequestHandler`
- 监控采集：`HardwareMonitor`

运行流程：

1. 浏览器访问 `/`，返回 `HTML_TEMPLATE`。
2. 前端通过 `/api/config` 拉取当前配置并渲染表单。
3. 用户点击“保存配置”或“保存并启动”。
4. 后端保存 `config.json`，若启动则由 `RunManager` 拉起 `do.py`。
5. 前端每 1.5s 轮询 `/api/run/status`，增量拉取日志与状态。

### 2.2 任务分发衔接

WebUI 不直接调用 task 文件，而是统一走：

`webui.py` -> `do.py` -> `utils/control/protect.py` -> `utils/task/task{1,2,3}.py`

与 CLI 路径一致，保证行为一致性。

---

## 3. 前端 UI 关键实现

### 3.1 配置驱动表单

- 全局字段由 `FORM_SCHEMA["global"]` 定义。
- 各任务字段由 `FORM_SCHEMA["tasks"]["1"|"2"|"3"]` 定义。
- Task2 中 `COMPARE_MODE` 为下拉选择：`pta_msa` / `pta_mf`。
- Task2 额外暴露了：`MF_ENV`、`MF_ARGS_PATH`、`ENABLE_MF_WEIGHT_LOAD`。

### 3.4 模式感知配置

- 当前选中任务若为 `COMPARE_MODE=pta_mf`，全局 MSA 字段（`MSA_NAME`、`MSA_PATH`）会在 UI 中自动隐藏且禁用。
- 当前选中任务若为 `COMPARE_MODE=pta_msa`，MF 相关字段会自动隐藏：
   - 全局 `MF_NAME`
   - Task1 的 `SUPPORT_MF`
   - Task2 的 `MF_ENV`、`MF_ARGS_PATH`、`ENABLE_MF_WEIGHT_LOAD`
- 模式切换后会自动恢复对应字段显示：
   - 切回 `pta_msa` 时恢复 MSA 字段；
   - 切回 `pta_mf` 时恢复 MF 相关字段。
- 后端 `normalize_config` 与前端行为一致：
   - 选中任务是 `pta_mf` 时，MSA 字段允许为空；
    - 选中任务是 `pta_msa` 时，MSA 字段必须非空；
    - 选中任务是 `pta_msa` 时，MF 相关字段允许为空或被忽略。

### 3.2 运行态可视化

- 运行状态卡片展示 `state / pid / elapsed / output_dir / returncode`。
- 日志窗口支持增量追加与自动滚动。
- 结果面板可刷新 `output` 列表并打开 `report.html`、`log.txt`、`config.json`。
- 文件浏览模式下，支持对单个 `iters/iter_x/` 目录一键打包下载为 ZIP，便于导出某轮全部资料。
- 支持对历史 output 执行“重新生成 analysis”。

### 3.3 硬件信息

- 通过 `HardwareMonitor` 采集 CPU、内存、NPU。
- NPU 优先使用 `npu-smi info` 解析。
- 前端提供展开/收起开关，不影响任务执行。

---

## 4. 后端 API 设计

### 4.1 配置接口

- `GET /api/config`：返回当前有效配置（已与默认值合并）。
- `POST /api/config`：保存配置并做严格校验。

校验核心在 `normalize_config`，包含：

- `task_type` 范围校验（1~3）
- Task2 的 `MODELS` / `SUBMODULES` 长度一一对应校验
- `COMPARE_MODE` 选项校验（`pta_msa`、`pta_mf`）

### 4.2 运行控制接口

- `POST /api/run/start`：保存配置后启动 `do.py`
- `GET /api/run/status?cursor=x`：返回状态与增量日志
- `POST /api/run/stop`：发送停止信号（`terminate`）

### 4.3 结果接口

- `GET /api/results`：列举 `output/*` 并给出可视化入口
- `GET /api/results/iter-archive?output_id=<id>&path=iters/iter_x`：动态打包并下载某个迭代目录
- `POST /api/results/analysis/regenerate`：重建指定 output 的 analysis
- `GET /results/<output>/<file>`：读取历史产物文件

---

## 5. Task2 pta_mf 路径说明

Task2 在 `utils/task/task2.py` 中支持 `COMPARE_MODE`：

- `pta_msa`：PTA + MSA
- `pta_mf`：PTA + MF

当 `COMPARE_MODE=pta_mf` 时：

1. 执行 PTA mutate。
2. 执行 PTA SAVE，生成共享权重。
3. 执行 PTA LOAD 校验。
4. 进入 MF 分支：
   - 若 `ENABLE_MF_WEIGHT_LOAD=true`，先做 pth->npz->ckpt 转换；
   - 再执行 MF 加载验证。
5. 产出 `status.json`、`runtime_logs/`、`scripts/` 等迭代证据。
