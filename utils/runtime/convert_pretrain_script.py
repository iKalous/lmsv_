#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本转换器：将pretrain_mutated_qwen2.sh转换为新的格式
保留GPT_ARGS、MLA_ARGS、MOE_ARGS、ROPE_ARGS、DATA_ARGS、OUTPUT_ARGS
新增：自动识别并保留原脚本中的分词器配置（--tokenizer-type、--tokenizer-model、--tokenizer-name-or-path）
修复：解决重复参数组、参数提取不完整、错误参数添加、冗余空定义等问题
"""

import re
import sys
import os
from collections import OrderedDict


def force_zero_dropout(args_content):
    """Force attention/hidden dropout flags to 0.0 in converted scripts."""
    if not args_content:
        return args_content
    args_content = re.sub(r'--attention-dropout\s+\S+', '--attention-dropout 0.0', args_content)
    args_content = re.sub(r'--hidden-dropout\s+\S+', '--hidden-dropout 0.0', args_content)
    return args_content


def sanitize_args_content(args_content):
    """
    清理异常提取结果，避免把相邻的 XXX_ARGS 变量头误当成参数内容。
    """
    if not args_content:
        return ""

    cleaned_lines = []
    for raw_line in args_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r'"?[A-Z_]+_ARGS=.*', line):
            continue
        if re.fullmatch(r'--"?', line):
            continue
        if '_ARGS=' in line:
            continue
        cleaned_lines.append(raw_line)

    return '\n'.join(cleaned_lines)

def extract_args_section(content, args_name):
    """
    提取指定的ARGS部分 - 修复空定义、续行符处理和结束位置判断
    处理规则：
    1. 支持空参数组定义（例如 MOE_ARGS=""）
    2. 支持多行续行符（\）的参数组
    3. 去重重复的参数组定义（取最后一次定义的内容）
    4. 清理多余的续行符
    """
    # 逐行解析，避免把 `FOO_ARGS=""` 后面的下一个参数组头误吞进去
    pattern = re.compile(rf'^{args_name}="', re.MULTILINE)
    matches = list(pattern.finditer(content))
    if not matches:
        return ""

    args_content = ""
    for match in matches:
        start = match.end()

        # 单行空定义或单行内容：NAME="..."。
        line_end = content.find('\n', start)
        if line_end == -1:
            line_end = len(content)
        same_line_suffix = content[start:line_end]
        if same_line_suffix.endswith('"'):
            args_content = same_line_suffix[:-1]
            continue

        # 多行定义：持续读取到仅包含一个双引号的结束行。
        pos = line_end + 1 if line_end < len(content) else len(content)
        collected = []
        while pos < len(content):
            next_line_end = content.find('\n', pos)
            if next_line_end == -1:
                next_line_end = len(content)
            line = content[pos:next_line_end]
            if '"' in line:
                prefix, _, suffix = line.partition('"')
                if prefix.strip():
                    collected.append(prefix)
                # 同一行出现结束引号，说明当前参数组到此结束；
                # 即使引号后还跟着下一个 XXX_ARGS 头，也不应继续吞入当前组。
                break
            if line.strip() == '"':
                break
            collected.append(line)
            pos = next_line_end + 1
        args_content = '\n'.join(collected)

    args_content = sanitize_args_content(args_content)

    # 清理续行符和多余空格
    args_content = re.sub(r'\\\n', '\n', args_content)  # 替换续行符为换行
    args_content = re.sub(r'\s+', ' ', args_content)    # 合并多余空格
    args_content = re.sub(r'^\s+|\s+$', '', args_content)  # 去首尾空格

    if not args_content:
        return ""

    # 按参数分行（保持格式整洁）
    args_lines = []
    for arg in args_content.split('--'):
        if arg.strip():
            args_lines.append(f'    --{arg.strip()}')

    return '\n'.join(args_lines)

def extract_variables(content):
    """
    提取简单的单行变量赋值，排除带引号的多行字符串变量
    修复：优化过滤条件，避免漏提关键变量
    """
    result = []
    # 要排除的多行变量名
    exclude_vars = {'DISTRIBUTED_ARGS', 'GPT_ARGS', 'MLA_ARGS', 'MOE_ARGS', 
                    'ROPE_ARGS', 'DATA_ARGS', 'OUTPUT_ARGS', 'DATA_CONFIG'}

    for line in content.split('\n'):
        line = line.strip()

        # 跳过空行、注释、续行符行
        if not line or line.startswith('#') or line.endswith('\\'):
            continue

        # 检查是否包含等号且是简单变量赋值
        if '=' in line:
            var_name, var_value = line.split('=', 1)
            var_name = var_name.strip()
            var_value = var_value.strip()

            # 过滤规则：全大写变量名 + 非排除列表 + 单行简单值
            if (var_name.isupper() and 
                var_name not in exclude_vars and
                not (var_value.startswith('"') and var_value.endswith('"')) and
                not (var_value.startswith("'") and var_value.endswith("'"))):
                result.append(line)

    return '\n'.join(result)

def extract_tokenizer_args(content):
    """
    从原脚本内容中提取所有分词器相关参数
    支持：--tokenizer-type、--tokenizer-model、--tokenizer-name-or-path
    """
    tokenizer_args = []
    tokenizer_pattern = r'(--tokenizer-(type|model|name-or-path)\s+[^\\\n]+)'
    
    # 先从所有ARGS部分中查找
    all_args_sections = [
        extract_args_section(content, "GPT_ARGS"),
        extract_args_section(content, "MLA_ARGS"),
        extract_args_section(content, "MOE_ARGS"),
        extract_args_section(content, "ROPE_ARGS"),
        extract_args_section(content, "DATA_ARGS"),
        extract_args_section(content, "OUTPUT_ARGS")
    ]
    
    # 合并所有ARGS内容进行查找
    all_args_content = '\n'.join(all_args_sections)
    matches = re.findall(tokenizer_pattern, all_args_content, re.IGNORECASE)
    
    # 处理匹配结果
    for match in matches:
        arg_line = match[0].strip()
        if arg_line:
            tokenizer_args.append(arg_line)
    
    # 如果没找到，直接从整个文件内容查找
    if not tokenizer_args:
        matches = re.findall(tokenizer_pattern, content, re.IGNORECASE)
        for match in matches:
            arg_line = match[0].strip()
            if arg_line:
                tokenizer_args.append(arg_line)
    
    # 去重并保持顺序
    seen = set()
    unique_tokenizer_args = []
    for arg in tokenizer_args:
        key = arg.split()[0]
        if key not in seen:
            seen.add(key)
            unique_tokenizer_args.append(arg)
    
    return unique_tokenizer_args

def convert_script(input_file, output_file=None):
    """转换脚本文件 - 修复重复参数、错误参数、冗余定义等问题"""
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return False
    
    # 读取原文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取各个ARGS部分（自动去重重复定义）
    args_map = OrderedDict([
        ("GPT_ARGS", force_zero_dropout(extract_args_section(content, "GPT_ARGS"))),
        ("MLA_ARGS", extract_args_section(content, "MLA_ARGS")),
        ("MOE_ARGS", extract_args_section(content, "MOE_ARGS")),
        ("ROPE_ARGS", extract_args_section(content, "ROPE_ARGS")),
        ("DATA_ARGS", extract_args_section(content, "DATA_ARGS")),
        ("OUTPUT_ARGS", extract_args_section(content, "OUTPUT_ARGS"))
    ])
    
    # 提取分词器相关参数
    tokenizer_args_list = extract_tokenizer_args(content)
    
    # 构建新脚本内容
    new_script = '''export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1
export NCCL_DETERMINISTIC=1
'''
    extracted_var = extract_variables(content)
    if extracted_var:
        new_script += extracted_var + "\n\n"
    
    # 构建DATA_CONFIG（移除错误的--save参数，仅保留必要内容）
    config_script = '''CKPT_LOAD_DIR="None"
DISTRIBUTED_ARGS="
    --master_addr $MASTER_ADDR \\
    --node_rank $NODE_RANK \\
    --worker_num $WORLD_SIZE \\
    --local_worker_num $NPUS_PER_NODE \\
    --master_port $MASTER_PORT \\
    --log_dir=msrun_log \\
    --join=False \\
    --cluster_time_out=300 \\
    --bind_core=True \\
"
DATA_CONFIG="
    --data-path $DATA_PATH \\
'''
    
    # 插入提取到的分词器参数
    if tokenizer_args_list:
        for tokenizer_arg in tokenizer_args_list:
            config_script += f'    {tokenizer_arg} \\\n'
    config_script = config_script.rstrip(' \\') + '\n"\n\n'  # 清理最后一个续行符
    new_script += config_script
    
    # 添加ARGS变量（仅添加非空的，避免冗余空定义）
    for args_name, args_content in args_map.items():
        if args_content:
            new_script += f'{args_name}="\n{args_content}\n"\n\n'
        else:
            # 空参数组不生成冗余定义，后续直接拼接空字符串
            pass
    
    # 添加尾部调用内容（兼容空参数组）
    new_script += '''msrun $DISTRIBUTED_ARGS pretrain_gpt.py \\
    $DATA_CONFIG \\
    ${GPT_ARGS:-} \\
    ${DATA_ARGS:-} \\
    ${OUTPUT_ARGS:-} \\
    ${MLA_ARGS:-} \\
    ${ROPE_ARGS:-} \\
    ${MOE_ARGS:-} \\
    --distributed-backend nccl \\
    --ai-framework mindspore
'''
    
    # 确定输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.sh"
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_script)
    
    print(f"转换完成！输出文件：{output_file}")
    if tokenizer_args_list:
        print(f"识别到的分词器配置：{tokenizer_args_list}")
    else:
        print("未识别到分词器配置，DATA_CONFIG中不添加分词器参数")
    return True

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python script_converter.py <输入文件> [输出文件]")
        print("示例: python script_converter.py pretrain_mutated_qwen2.sh")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 处理输出目录不存在的情况
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    success = convert_script(input_file, output_file)
    if success:
        print("脚本转换成功！")
    else:
        print("脚本转换失败！")

if __name__ == "__main__":
    main()
