import pandas as pd
import numpy as np

def merge_csv_files():
    
    # 读取CSV文件
    try:
        pta_df = pd.read_csv('res/submodule_execution_pta.csv') #submodule_execution_pta
        msa_df = pd.read_csv('res/submodule_execution_msa.csv') #submodule_execution_msa

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
        return None

    merged_df = pd.merge(pta_df, msa_df, on='Iteration', how='outer', suffixes=('_pta', '_msa'))

    result_data = []
    
    for _, row in merged_df.iterrows():
        iteration = row['Iteration']
        
        # 检查msa数据中是否有'-'
        msa_row = msa_df[msa_df['Iteration'] == iteration]
        has_dash = False
        
        if not msa_row.empty:
            for col in ['Execution Time (s)', 'NPU Memory (MB)', 'loss']:
                if msa_row[col].iloc[0] == '-':
                    has_dash = True
                    break
        
        if has_dash:
            result_row = {
                'Iteration': iteration,
                'Execution Time (s)_pta': '-',
                'NPU Memory (MB)_pta': '-',
                'Execution Time (s)_msa': '-',
                'NPU Memory (MB)_msa': '-',
                'loss_diff': '-',
                'Execution Time Diff (%)': '-',
                'NPU Memory Diff (%)': '-'
            }
        else:
            # 正常情况：复制数据并计算差值
            pta_loss = row['loss_pta']
            msa_loss = row['loss_msa']
            
            # 计算loss差值
            try:
                if (pd.notna(pta_loss) and pd.notna(msa_loss) and 
                    pta_loss != '-' and msa_loss != '-'):
                    loss_diff = round(abs(float(pta_loss) - float(msa_loss)), 4)
                else:
                    loss_diff = '-'
            except (ValueError, TypeError):
                loss_diff = '-'
            
            # 计算执行时间差异百分比 - 修改为 (大-小)/小
            try:
                if (pd.notna(row['Execution Time (s)_pta']) and pd.notna(row['Execution Time (s)_msa']) and
                    row['Execution Time (s)_pta'] != '-' and row['Execution Time (s)_msa'] != '-'):
                    time_pta = float(row['Execution Time (s)_pta'])
                    time_msa = float(row['Execution Time (s)_msa'])
                    # 修改为 (大-小)/小
                    if time_pta >= time_msa:
                        time_diff_pct = round(((time_pta - time_msa) / time_msa) * 100, 2)
                    else:
                        time_diff_pct = round(((time_msa - time_pta) / time_pta) * 100, 2)
                else:
                    time_diff_pct = '-'
            except (ValueError, TypeError):
                time_diff_pct = '-'
            
            # 计算内存使用差异百分比 - 修改为 (大-小)/小
            try:
                if (pd.notna(row['NPU Memory (MB)_pta']) and pd.notna(row['NPU Memory (MB)_msa']) and
                    row['NPU Memory (MB)_pta'] != '-' and row['NPU Memory (MB)_msa'] != '-'):
                    mem_pta = float(row['NPU Memory (MB)_pta'])
                    mem_msa = float(row['NPU Memory (MB)_msa'])
                    # 修改为 (大-小)/小
                    if mem_pta >= mem_msa:
                        mem_diff_pct = round(((mem_pta - mem_msa) / mem_msa) * 100, 2)
                    else:
                        mem_diff_pct = round(((mem_msa - mem_pta) / mem_pta) * 100, 2)
                else:
                    mem_diff_pct = '-'
            except (ValueError, TypeError):
                mem_diff_pct = '-'
            
            result_row = {
                'Iteration': iteration,
                'Execution Time (s)_pta': row['Execution Time (s)_pta'],
                'Execution Time (s)_msa': row['Execution Time (s)_msa'],
                'NPU Memory (MB)_pta': row['NPU Memory (MB)_pta'],
                'NPU Memory (MB)_msa': row['NPU Memory (MB)_msa'],
                'loss_diff': loss_diff,
                'Execution Time Diff (%)': time_diff_pct,
                'NPU Memory Diff (%)': mem_diff_pct
            }
        
        result_data.append(result_row)
    

    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values('Iteration').reset_index(drop=True)
    

    output_file = 'merged_execution.csv'
    result_df.to_csv(output_file, index=False)
    
    # 生成文本报告（包含总结）
    generate_text_report(result_df)

    return result_df

def generate_text_report(df):
    """生成文本报告并在最后添加总结"""
    # 过滤出有效数据
    valid_time_data = df[df['Execution Time Diff (%)'] != '-']
    valid_mem_data = df[df['NPU Memory Diff (%)'] != '-']
    valid_loss_data = df[df['loss_diff'] != '-']
    
    # 计算统计信息
    time_avg_diff = valid_time_data['Execution Time Diff (%)'].astype(float).mean() if not valid_time_data.empty else 0
    mem_avg_diff = valid_mem_data['NPU Memory Diff (%)'].astype(float).mean() if not valid_mem_data.empty else 0
    loss_avg_diff = valid_loss_data['loss_diff'].astype(float).mean() if not valid_loss_data.empty else 0
    
    # 确定哪个框架性能更好（基于原始值比较，而不是百分比）
    # 需要重新计算原始数据的平均值来正确比较性能
    valid_time_pta = df[df['Execution Time (s)_pta'] != '-']['Execution Time (s)_pta'].astype(float)
    valid_time_msa = df[df['Execution Time (s)_msa'] != '-']['Execution Time (s)_msa'].astype(float)
    valid_mem_pta = df[df['NPU Memory (MB)_pta'] != '-']['NPU Memory (MB)_pta'].astype(float)
    valid_mem_msa = df[df['NPU Memory (MB)_msa'] != '-']['NPU Memory (MB)_msa'].astype(float)
    
    avg_time_pta = valid_time_pta.mean() if not valid_time_pta.empty else 0
    avg_time_msa = valid_time_msa.mean() if not valid_time_msa.empty else 0
    avg_mem_pta = valid_mem_pta.mean() if not valid_mem_pta.empty else 0
    avg_mem_msa = valid_mem_msa.mean() if not valid_mem_msa.empty else 0
    
    time_comparison = "PTA" if avg_time_pta > avg_time_msa else "MSA"
    mem_comparison = "PTA" if avg_mem_pta > avg_mem_msa else "MSA"
    
    with open('comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PTA vs MSA 执行性能对比报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("详细数据对比:\n")
        f.write("-" * 80 + "\n")
        
        # 写入表头
        headers = ["Iter", "PTA Time(s)", "MSA Time(s)", "Time Diff(%)", 
                  "PTA Mem(MB)", "MSA Mem(MB)", "Mem Diff(%)", "Loss Diff"]
        f.write(f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} "
                f"{headers[4]:<12} {headers[5]:<12} {headers[6]:<12} {headers[7]:<10}\n")
        f.write("-" * 80 + "\n")
        
        # 写入每一行数据
        for _, row in df.iterrows():
            f.write(f"{row['Iteration']:<6} "
                   f"{str(row['Execution Time (s)_pta']):<12} "
                   f"{str(row['Execution Time (s)_msa']):<12} "
                   f"{str(row['Execution Time Diff (%)']):<12} "
                   f"{str(row['NPU Memory (MB)_pta']):<12} "
                   f"{str(row['NPU Memory (MB)_msa']):<12} "
                   f"{str(row['NPU Memory Diff (%)']):<12} "
                   f"{str(row['loss_diff']):<10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("性能对比总结:\n")
        f.write("=" * 80 + "\n\n")
        
        # 添加总结（基于原始平均值）
        if avg_time_pta > avg_time_msa:
            f.write(f"执行时间(PTA比MSA慢): {abs(avg_time_pta - avg_time_msa):.2f}s ({time_avg_diff:.2f}%)\n")
        else:
            f.write(f"执行时间(MSA比PTA慢): {abs(avg_time_msa - avg_time_pta):.2f}s ({time_avg_diff:.2f}%)\n")
            
        if avg_mem_pta > avg_mem_msa:
            f.write(f"显存占用(PTA比MSA多): {abs(avg_mem_pta - avg_mem_msa):.2f}MB ({mem_avg_diff:.2f}%)\n")
        else:
            f.write(f"显存占用(MSA比PTA多): {abs(avg_mem_msa - avg_mem_pta):.2f}MB ({mem_avg_diff:.2f}%)\n")
        
        f.write(f"平均loss差异: {loss_avg_diff:.4f}\n")
        

if __name__ == "__main__":
    result = merge_csv_files()
    if result is not None:
        print("分析完成！")
        print("生成的文件:")
        print("- comparison_report.txt")
    else:
        print("分析失败！")