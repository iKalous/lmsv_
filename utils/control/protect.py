import signal

import utils


def _protect_log(tag, message):
    text = str(message)
    if tag:
        return f"[守护模块][{tag}] {text}"
    return f"[守护模块] {text}"


def _ignore_sigterm():
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except ValueError:
        pass


def task(task_type, params):
    utils.log.write.info(_protect_log("任务", f"任务类型: {task_type}"))
    utils.log.write.info(_protect_log("任务", f"任务参数: {params}"))
    _ignore_sigterm()
    if task_type == 1:
        from utils.task import task1
        utils.task.task1.main(params)
    elif task_type == 2:
        from utils.task import task2
        utils.task.task2.main(params)
    elif task_type == 3:
        from utils.task import task3
        utils.task.task3.main(params)
    elif task_type == 4:
        from utils.task import task4
        utils.task.task4.main(params)
    elif task_type == 5:
        from utils.task import task5
        utils.task.task5.main(params)
    elif task_type == 6:
        from utils.task import task6
        utils.task.task6.main(params)
    else:
        utils.log.write.error(_protect_log("任务", f"任务类型错误: Unknown task_type: {task_type}"))
        raise ValueError(f"任务类型错误：Unknown task_type: {task_type}")
