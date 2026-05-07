#!/usr/bin/env python3
"""
bashToYaml.py - Convert Megatron bash training script to flat YAML format
进行两遍扫描，第二遍处理变量替换
"""

import re
import yaml
import sys
import os
from typing import Dict, Any, List, Union
import argparse


class BashToYamlConverter:
    """将 Megatron bash 训练脚本转换为扁平 YAML 格式的转换器"""
    
    def __init__(self):
        self.parameters = {}
        self.variables = {}  # 存储变量名和值的映射
        self.warnings = []
    
    def parse_bash_script(self, bash_content: str) -> Dict[str, Any]:
        """
        解析 bash 脚本内容并提取所有参数
        
        Args:
            bash_content: bash 脚本内容
            
        Returns:
            包含所有参数的字典，所有参数都在 model 字段下
        """
        self.parameters = {"model": {}}
        self.variables = {}
        
        # 第一遍扫描：收集所有变量
        self._first_pass(bash_content)
        
        # 第二遍扫描：解析参数并进行变量替换
        self._second_pass(bash_content)
        
        return self.parameters
    
    def _first_pass(self, bash_content: str):
        """第一遍扫描：收集所有变量定义"""
        lines = bash_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 只处理变量赋值语句
            if '=' in line and not line.startswith('--'):
                self._parse_variable_for_first_pass(line, line_num)
    
    def _parse_variable_for_first_pass(self, line: str, line_num: int):
        """第一遍扫描：解析变量赋值并存储到变量字典"""
        try:
            # 分割变量名和值
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 移除 export 关键字
                if key.startswith('export '):
                    key = key[7:].strip()
                
                # 处理特殊值
                # if value in ['None', 'null']:
                #     return
                
                # 移除引号
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # 处理多引号情况
                if value.startswith("'''") and value.endswith("'''"):
                    value = value[3:-3]
                
                # 转换值类型
                converted_value = self._convert_value_type(value)
                
                # 存储到变量字典
                self.variables[key] = converted_value
                
        except Exception as e:
            self.warnings.append(f"First pass - Line {line_num}: Failed to parse variable: {line} - {str(e)}")
    
    def _second_pass(self, bash_content: str):
        """第二遍扫描：解析所有参数并进行变量替换"""
        lines = bash_content.split('\n')
        current_section = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # 处理变量赋值（第二遍也需要处理，因为可能有新的变量）
            if '=' in line and not line.startswith('--'):
                self._parse_variable_assignment(line, line_num)
            
            # 处理命令行参数
            elif line.startswith('--'):
                self._parse_command_line_arg(line, line_num)
            
            # 处理以 torchrun 开头的行
            elif line.startswith('torchrun'):
                self._parse_torchrun_command(line, line_num)
    
    def _parse_variable_assignment(self, line: str, line_num: int):
        """第二遍扫描：解析变量赋值语句"""
        try:
            # 分割变量名和值
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 移除 export 关键字
                if key.startswith('export '):
                    key = key[7:].strip()
                
                # 处理特殊值
                if value in ['None', 'null']:
                    return
                
                # 移除引号
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # 处理多引号情况
                if value.startswith("'''") and value.endswith("'''"):
                    value = value[3:-3]
                
                # 进行变量替换
                value = self._replace_variables(value)
                
                # 转换值类型
                converted_value = self._convert_value_type(value)
                
                # 添加到参数中
                self.parameters["model"][key] = converted_value
                
                # 同时更新变量字典（第二遍扫描中可能遇到新的变量）
                self.variables[key] = converted_value
                
        except Exception as e:
            self.warnings.append(f"Second pass - Line {line_num}: Failed to parse variable assignment: {line} - {str(e)}")
    
    def _parse_command_line_arg(self, line: str, line_num: int):
        """第二遍扫描：解析命令行参数"""
        try:
            # 移除行尾的反斜杠
            if line.endswith('\\'):
                line = line[:-1].strip()
            
            parts = line.split()
            
            if len(parts) == 1:
                # 布尔标志，如 --use-flash-attn
                key = parts[0].lstrip('-')
                self.parameters["model"][key] = True
                
            elif len(parts) >= 2:
                # 键值对，如 --num-layers 32
                key = parts[0].lstrip('-')
                value = ' '.join(parts[1:])
                
                # 移除引号
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # 进行变量替换
                value = self._replace_variables(value)
                
                # 转换值类型
                converted_value = self._convert_value_type(value)
                self.parameters["model"][key] = converted_value
                
        except Exception as e:
            self.warnings.append(f"Second pass - Line {line_num}: Failed to parse command line argument: {line} - {str(e)}")
    
    def _parse_torchrun_command(self, line: str, line_num: int):
        """第二遍扫描：解析 torchrun 命令"""
        try:
            # 移除 torchrun 和脚本名，只处理参数
            parts = line.split()
            
            # 找到 pretrain_gpt.py 的位置
            script_index = -1
            for i, part in enumerate(parts):
                if 'pretrain_gpt.py' in part:
                    script_index = i
                    break
            
            if script_index != -1:
                # 处理 pretrain_gpt.py 后面的所有参数
                i = script_index + 1
                while i < len(parts):
                    arg = parts[i]
                    if arg.startswith('--'):
                        # 查找这个参数的值
                        if i + 1 < len(parts) and not parts[i + 1].startswith('--'):
                            key = arg.lstrip('-')
                            value = parts[i + 1]
                            
                            # 移除引号
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                value = value[1:-1]
                            
                            # 进行变量替换
                            value = self._replace_variables(value)
                            
                            converted_value = self._convert_value_type(value)
                            self.parameters["model"][key] = converted_value
                            i += 2  # 跳过参数值
                        else:
                            # 布尔标志
                            key = arg.lstrip('-')
                            self.parameters["model"][key] = True
                            i += 1
                    else:
                        i += 1
                        
        except Exception as e:
            self.warnings.append(f"Second pass - Line {line_num}: Failed to parse torchrun command: {line} - {str(e)}")
    
    def _replace_variables(self, value: str) -> str:
        """
        替换字符串中的变量引用 ${var_name} 和 $var_name
        
        Args:
            value: 可能包含变量引用的字符串
            
        Returns:
            替换变量后的字符串
        """
        if not isinstance(value, str):
            return value
        
        # 使用正则表达式匹配 ${var_name} 和 $var_name 格式
        patterns = [r'\$\{(\w+)\}', r'\$(\w+)']
        
        def replace_match(match):
            var_name = match.group(1)
            if var_name in self.variables:
                return str(self.variables[var_name])
            else:
                self.warnings.append(f"Variable ${var_name} not found in variable dictionary")
                return match.group(0)  # 保持原样
        
        # 替换所有匹配的变量引用
        for pattern in patterns:
            value = re.sub(pattern, replace_match, value)
        
        return value

    def _convert_value_type(self, value: str) -> Union[str, int, float, bool]:
        """转换值的类型"""
        if not isinstance(value, str):
            return value
        
        value = value.strip()
        
        # 处理布尔值
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # 处理数字
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        
        # 处理浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 处理变量引用（如 $VAR）
        if value.startswith('$'):
            return value
        
        # 其他情况保持为字符串
        return value
    
    def save_yaml(self, output_path: str):
        """保存为 YAML 文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.parameters, f, default_flow_style=False, 
                         allow_unicode=True, indent=2, sort_keys=False)
            print(f"✅ YAML configuration saved to: {output_path}")
            return self.parameters
        except Exception as e:
            print(f"❌ Error saving YAML file: {str(e)}")
    
    def print_warnings(self):
        """打印警告信息"""
        if self.warnings:
            print("\n⚠️  Conversion Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
            print()
    
    def print_variables(self):
        """打印收集到的变量（用于调试）"""
        if self.variables:
            print("\n📋 Collected Variables:")
            for key, value in self.variables.items():
                print(f"  {key} = {value}")
            print()

def bashtoyaml(input_file, output, show_variables=False, no_warnings=False):
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    # 读取 bash 脚本
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            bash_content = f.read()
    except Exception as e:
        print(f"❌ Error reading input file: {str(e)}")
        sys.exit(1)
    
    # 转换
    converter = BashToYamlConverter()
    parameters = converter.parse_bash_script(bash_content)
    
    # 保存 YAML
    result = converter.save_yaml(output)
    
    # 显示变量（调试用）
    if show_variables:
        converter.print_variables()
    
    # 显示警告
    if not no_warnings:
        converter.print_warnings()
    
    # 显示统计信息
    param_count = len(parameters.get("model", {}))
    var_count = len(converter.variables)
    print(f"📊 Converted {param_count} parameters, collected {var_count} variables")
    return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Convert Megatron bash script to flat YAML format')
    parser.add_argument('input_file', help='Input bash script file')
    parser.add_argument('-o', '--output', help='Output YAML file (default: output.yaml)', 
                       default='output.yaml')
    parser.add_argument('--no-warnings', action='store_true', 
                       help='Suppress warning messages')
    parser.add_argument('--show-variables', action='store_true',
                       help='Show collected variables for debugging')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # 读取 bash 脚本
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            bash_content = f.read()
    except Exception as e:
        print(f"❌ Error reading input file: {str(e)}")
        sys.exit(1)
    
    # 转换
    converter = BashToYamlConverter()
    parameters = converter.parse_bash_script(bash_content)
    
    # 保存 YAML
    converter.save_yaml(args.output)
    
    # 显示变量（调试用）
    if args.show_variables:
        converter.print_variables()
    
    # 显示警告
    if not args.no_warnings:
        converter.print_warnings()
    
    # 显示统计信息
    param_count = len(parameters.get("model", {}))
    var_count = len(converter.variables)
    print(f"📊 Converted {param_count} parameters, collected {var_count} variables")


if __name__ == "__main__":
    main()