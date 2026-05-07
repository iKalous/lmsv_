# Task 结果分析器架构

## 范围

本文档描述 `utils/analyze/task1_result.py` 中分析流水线的整体架构。
覆盖 task1/task2/task3，重点说明 task1 在 `pta_msa` 与 `pta_mf` 两种模式下的统一实现。

## 入口与产物

- 主入口：`analyze_task_run(output_root, run_dir, model_name, planned_iterations, task_type)`
- 对外包装：`analyze_task1_run`、`analyze_task2_run`、`analyze_task3_run`

在 `output/<run>/analysis` 下生成：

- `data/summary.json`：结构化汇总数据
- `data/iteration_metrics.csv`：逐轮指标表
- `data/issue_groups.json`：问题分组结果
- `summary.md`：Markdown 报告
- `report.html`：HTML 报告
- `assets/*.svg`：性能/精度/显存图表

每轮也会生成：

- `iters/iter_x/report.md`：该轮的简版分析报告

## 核心数据模型

核心有两个 dataclass：

- `IterationAnalysis`：单轮运行状态、指标差值、路径信息、步骤结果
- `AnalysisArtifacts`：整体分析产物路径、统计计数与输出索引

`IterationAnalysis` 使用 `msa_*` 字段承载“对比侧”统一槽位。
当 task1 为 `pta_mf` 时，会将 MF 的 csv 映射到这个对比槽位，从而复用下游指标计算与渲染逻辑。

## 流水线阶段

1. 发现并标准化运行上下文

- 解析 run 目录并推断任务类型
- 从 `output/config.json` 读取任务配置
- 解析 `COMPARE_MODE`

2. 采集迭代材料

- 定位当前轮 csv 与 step csv
- 读取 `status.json` 与运行日志
- 收集 mutation 输入与备份材料

3. 构建对比指标

- 序列模式：`_compare_series_csvs`
- 单行模式：`_compare_single_iteration`
- 可选 step 级覆盖：`_override_performance_metrics`

4. 推导执行状态与问题类型

- 从 `status.json` 抽取组件成功/失败状态
- 通过 `_derive_functional_reasons` 归纳功能失败原因
- 通过 `_scan_log_for_signals` 提取日志关键信号
- 通过 `_categorize_iteration` 计算问题分类与总体状态

5. 渲染报告

- 单轮 Markdown：`_render_iteration_report`
- 全局输出：`_summary_payload`、`_write_markdown`、`_write_html_report`
- 图表输出：`_write_svg_bar_chart`

## 对比模式统一

### 目标

在一个分析框架内同时支持 task1 两种对比模式：

- `pta_msa`：PTA 对比 MSA
- `pta_mf`：PTA 对比 MF

### 关键设计

1. 对比标签抽象

- `_compare_label_for_mode(task_type, compare_mode)` 返回当前对比对象（task1 下为 `MSA` 或 `MF`）

2. 数据源重映射

- 在 task1 + `pta_mf` 下：
  - `msa_csv = mf_csv`
  - `msa_step_csv = mf_step_csv or mf_csv`

这样可使指标函数继续使用 `pta` 与对比槽位（`msa`）进行计算，避免分支复制。

3. 文案动态渲染

- 单轮报告、summary markdown、html 报告都按对比对象动态渲染
- 例如在 `pta_mf` 下展示 `PTA/MF` 与 `MF执行成功率`

4. 向后兼容

- 保留已有字段：`MS执行成功数`、`MS执行成功率`
- 新增元数据字段：
  - `compare_mode`
  - `对比对象`

## 指标语义

性能：

- `pta_avg_step_time_skip1` 与对比侧平均 step 耗时
- 差值：`compare - pta`
- 比例：`(compare - pta) / pta`

精度：

- `max_loss_diff`：对齐 step 上的最大 loss 绝对差

显存：

- 对比侧最大显存减去 pta 最大显存
- 比例相对 pta 计算

等级由 `_severity` 结合 `utils/analyze/rules.py` 中阈值生成。

## 失败与问题分类

- 执行失败由组件状态、关键产物缺失与失败标记共同驱动
- 功能问题由显式组件失败与回退材料检查共同判定
- 精度/性能/显存问题基于指标阈值判定

最终状态映射：

- `MUTATION_FAILED`
- `EXECUTION_FAILED`
- `COMPLETED_WITH_ISSUES`
- `PASS`

## 扩展指引

新增对比模式或任务变体时建议：

1. 在 `TASK_PROFILES` 扩展任务配置
2. 提供 csv 路径映射与对比模式定义
3. 需要时补充对比标签映射
4. 将模式差异收敛在“采集/映射阶段”，避免复制指标与渲染逻辑
5. 对已知 output 重新执行分析并核验：
   - `iters/iter_x/report.md`
   - `analysis/summary.md`
   - `analysis/report.html`
   - `analysis/data/summary.json`
