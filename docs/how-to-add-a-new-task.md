### 配置层面
修改 getconf.py:
1. 在 `config.json.example`中添加对应任务的专属配置样例；
2. 修改 `ask_task_type()` 函数，添加对应的任务编号与描述；
3. 修改 `build_task_config()`函数，添加对应任务的专属配置;
4. 修改文件头的 `TASK_DESC`

### 任务内容
1. 修改 `do.py`中的`TASK_LABELS`
2. 新增 taskx.py
3. 在 `utils/task/__init__.py` 中新增taskx
4. 在 `control/protect.py`中新增分支

### 结果分析阶段
新增 `taskx_result.py`

修改 `utils/analyze/manual.py`:
1. 在 `regenerate_output_analysis` 函数中添加对应任务的专属分析逻辑;

修改 `utils/analyze/task1_result.py`:
1. 在 `TASK_PROFILES` 中添加对应任务的专属配置;

在自己的taskx.py中添加分析逻辑:
