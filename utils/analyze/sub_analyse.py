import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="分析PTA和MSA验证日志")
    parser.add_argument(
        "--pta_log",
        type=str,
        default="res/submodule_execution_pta.csv",
        help="PTA日志文件名",
    )
    parser.add_argument(
        "--msa_log",
        type=str,
        default="res/submodule_execution_msa.csv",
        help="MSA日志文件名",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="res/analyse_report/recomparison_report.txt",
        help="输出报告文件路径",
    )
    parser.add_argument(
        "--peer-label",
        type=str,
        default="MSA",
        help="对端名称（如 MSA 或 MF），用于报告文案显示",
    )
    return parser.parse_args()


def calc_relative_diff(left, right):
    import pandas as pd

    try:
        if pd.notna(left) and pd.notna(right) and left != "-" and right != "-":
            left = float(left)
            right = float(right)
            base = min(left, right)
            if base == 0:
                return "-"
            return round(((max(left, right) - base) / base) * 100, 2)
    except (ValueError, TypeError):
        return "-"
    return "-"


def calc_loss_diff(pta_loss, msa_loss):
    import pandas as pd

    try:
        if pd.notna(pta_loss) and pd.notna(msa_loss) and pta_loss != "-" and msa_loss != "-":
            return round(abs(float(pta_loss) - float(msa_loss)), 4)
    except (ValueError, TypeError):
        return "-"
    return "-"


def merge_csv_files(pta_log_path, msa_log_path, report_path):
    import pandas as pd

    try:
        pta_df = pd.read_csv(pta_log_path)
        msa_df = pd.read_csv(msa_log_path)
        pta_df["Iteration"] = pta_df["Iteration"].astype(int)
        msa_df["Iteration"] = msa_df["Iteration"].astype(int)
    except FileNotFoundError as exc:
        print(f"错误：找不到文件 - {exc}")
        return None

    merged_df = pd.merge(pta_df, msa_df, on="Iteration", how="outer", suffixes=("_pta", "_msa"))
    result_data = []

    for _, row in merged_df.iterrows():
        iteration = row["Iteration"]
        msa_row = msa_df[msa_df["Iteration"] == iteration]
        has_dash = (
            not msa_row.empty
            and any(msa_row[col].iloc[0] == "-" for col in ["Execution Time (s)", "NPU Memory (MB)", "loss"])
        )

        if has_dash:
            result_data.append(
                {
                    "Iteration": iteration,
                    "Execution Time (s)_pta": "-",
                    "NPU Memory (MB)_pta": "-",
                    "Execution Time (s)_msa": "-",
                    "NPU Memory (MB)_msa": "-",
                    "loss_diff": "-",
                    "Execution Time Diff (%)": "-",
                    "NPU Memory Diff (%)": "-",
                }
            )
            continue

        result_data.append(
            {
                "Iteration": iteration,
                "Execution Time (s)_pta": row["Execution Time (s)_pta"],
                "Execution Time (s)_msa": row["Execution Time (s)_msa"],
                "NPU Memory (MB)_pta": row["NPU Memory (MB)_pta"],
                "NPU Memory (MB)_msa": row["NPU Memory (MB)_msa"],
                "loss_diff": calc_loss_diff(row["loss_pta"], row["loss_msa"]),
                "Execution Time Diff (%)": calc_relative_diff(
                    row["Execution Time (s)_pta"],
                    row["Execution Time (s)_msa"],
                ),
                "NPU Memory Diff (%)": calc_relative_diff(
                    row["NPU Memory (MB)_pta"],
                    row["NPU Memory (MB)_msa"],
                ),
            }
        )

    result_df = pd.DataFrame(result_data).sort_values("Iteration").reset_index(drop=True)
    result_df.to_csv("merged_execution.csv", index=False)
    return result_df


def generate_text_report(df, report_file, peer_label="MSA"):
    valid_time_data = df[df["Execution Time Diff (%)"] != "-"]
    valid_mem_data = df[df["NPU Memory Diff (%)"] != "-"]
    valid_loss_data = df[df["loss_diff"] != "-"]

    successful_models_count = len(valid_loss_data)
    time_within_threshold = valid_time_data[valid_time_data["Execution Time Diff (%)"].astype(float) <= 20].shape[0]
    mem_within_threshold = valid_mem_data[valid_mem_data["NPU Memory Diff (%)"].astype(float) <= 20].shape[0]
    loss_within_threshold = valid_loss_data[valid_loss_data["loss_diff"].astype(float) <= 1e-3].shape[0]

    with open(report_file, "w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write(f"PTA vs {peer_label} 执行性能对比报告\n")
        handle.write("=" * 80 + "\n\n")
        handle.write("详细数据对比:\n")
        handle.write("-" * 80 + "\n")

        headers = [
            "Iter",
            "PTA Time(s)",
            f"{peer_label} Time(s)",
            "Time Diff(%)",
            "PTA Mem(MB)",
            f"{peer_label} Mem(MB)",
            "Mem Diff(%)",
            "Loss Diff",
        ]
        handle.write(
            f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} "
            f"{headers[4]:<12} {headers[5]:<12} {headers[6]:<12} {headers[7]:<10}\n"
        )
        handle.write("-" * 80 + "\n")

        for _, row in df.iterrows():
            handle.write(
                f"{int(row['Iteration']):<6d} "
                f"{str(row['Execution Time (s)_pta']):<12} "
                f"{str(row['Execution Time (s)_msa']):<12} "
                f"{str(row['Execution Time Diff (%)']):<12} "
                f"{str(row['NPU Memory (MB)_pta']):<12} "
                f"{str(row['NPU Memory (MB)_msa']):<12} "
                f"{str(row['NPU Memory Diff (%)']):<12} "
                f"{str(row['loss_diff']):<10}\n"
            )

        handle.write("\n" + "=" * 80 + "\n")
        handle.write("性能对比总结:\n")
        handle.write("=" * 80 + "\n\n")
        handle.write(f"验证成功的模型数量: {successful_models_count}\n")
        handle.write(f"执行时间(差异在20%以内的数量): {time_within_threshold}\n")
        handle.write(f"显存占用(差异在20%以内的数量): {mem_within_threshold}\n")
        handle.write(f"loss(差异在1e-4以内的数量): {loss_within_threshold}\n")


def main():
    args = parse_args()
    os.makedirs("res/analyse_report", exist_ok=True)
    print(f"使用PTA日志文件: {args.pta_log}")
    print(f"使用{args.peer_label}日志文件: {args.msa_log}")

    result = merge_csv_files(args.pta_log, args.msa_log, args.report)
    if result is None:
        print("分析失败！")
        return 1

    generate_text_report(result, args.report, peer_label=args.peer_label)

    print("分析完成！")
    print("生成的文件:")
    print(f"res/analyse_report/{os.path.basename(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
