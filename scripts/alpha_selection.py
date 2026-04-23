#!/usr/bin/env python3
"""
α 衰减因子选择

固定测试模型：NB + ZINB（共享流程，双模型）。
对候选 α 重构网络权重、提取特征、拟合 NB 与 ZINB，以 ZINB 的 Log-Likelihood 选出最优 α。

支持两种步骤：
- Step2.1: 自变量不含 Betweenness（用于 HDKN 回归前的 α 选择）
- Step3:   自变量含 Betweenness（用于 ACO 前的 α 选择）

运行方式：
    python scripts/alpha_selection.py --run-id full --step 2.1
    python scripts/alpha_selection.py --run-id full --step 3
    python scripts/alpha_selection.py --run-id full --step 2.1 --alphas 0.85,0.90,0.95,1.0

可复现性：为保证多次运行 LL 一致，建议使用 PYTHONHASHSEED=42：
    PYTHONHASHSEED=42 python scripts/alpha_selection.py --run-id full --step 2.1 --alphas 0.6,0.8,1.0
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from scripts.common import init_script, parse_comma_list
init_script()

from loguru import logger

from scripts.common import run_alpha_selection

# Step2.1: HDKN 特征（不含 Betweenness、Avg_Weight），控制变量 Back_cite, Assignee, Total_pat
X_VARS_2_1 = ["New_n", "New_e", "Min_pn", "Con_n", "Con_e", "Eigen", "Constraint"]
# Step3: 不含 Betweenness、Avg_Weight（二者不计算，以节省时间）
X_VARS_3 = ["New_n", "New_e", "Min_pn", "Con_n", "Con_e", "Constraint", "Eigen"]

STEP_CONFIG = {
    "2.1": {
        "step_name": "02_1_alpha_selection",
        "step_label": "Step2.1",
        "output_subdir": "02_1_alpha_selection",
        "x_vars": X_VARS_2_1,
    },
    "3": {
        "step_name": "03_alpha_selection",
        "step_label": "Step3",
        "output_subdir": "03_alpha_selection",
        "x_vars": X_VARS_3,
        "report_title": "α 衰减因子选择报告",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="α 衰减因子选择（Step2.1 或 Step3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/alpha_selection.py --run-id full --step 2.1
  python scripts/alpha_selection.py --run-id full --step 3
  python scripts/alpha_selection.py --run-id full --step 2.1 --alphas 0.80,0.85,0.90,0.95,0.99,1.00
        """,
    )
    parser.add_argument("--run-id", type=str, required=True, help="运行 ID（来自 Step1）")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["2.1", "3"],
        help="步骤：2.1=HDKN 回归前 α 选择，3=ACO 前 α 选择",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="候选 α 列表，逗号分隔，如 0.85,0.90,0.95,1.0",
    )
    parser.add_argument("--patents-csv", type=Path, default=None, help="专利数据 CSV 路径")
    parser.add_argument("--networks-dir", type=Path, default=None, help="网络目录路径")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="自定义自变量，逗号分隔，如 New_e,Con_e,Constraint（覆盖默认 X_VARS）",
    )
    parser.add_argument("--force", action="store_true", help="强制重算")
    parser.add_argument(
        "--min-app-year",
        type=int,
        default=None,
        help="最小申请年份，与 Step1 一致（如 2012）",
    )
    parser.add_argument(
        "--no-control-vars",
        action="store_true",
        help="不包含控制变量（Back_cite, Assignee, Total_pat）",
    )
    args = parser.parse_args()

    cfg = STEP_CONFIG[args.step]
    x_vars = cfg["x_vars"]
    if args.features:
        x_vars = [s.strip() for s in args.features.split(",") if s.strip()]
    alphas = parse_comma_list(args.alphas)
    if alphas:
        alphas = [float(x) for x in alphas]

    include_control_vars = not args.no_control_vars
    output_subdir = cfg["output_subdir"]
    if args.no_control_vars:
        output_subdir = output_subdir.rstrip("/") + "_no_control"
    if args.features:
        feat_suffix = "_".join(s.strip().replace(" ", "") for s in args.features.split(","))
        output_subdir = output_subdir.rstrip("/") + "_" + feat_suffix
    kwargs = {
        "run_id": args.run_id,
        "step_name": cfg["step_name"],
        "step_label": cfg["step_label"],
        "output_subdir": output_subdir,
        "x_vars": x_vars,
        "alphas": alphas if alphas else None,
        "patents_csv": args.patents_csv,
        "networks_dir": args.networks_dir,
        "force": args.force,
        "min_app_year": args.min_app_year,
        "include_control_vars": include_control_vars,
    }
    if "report_title" in cfg:
        kwargs["report_title"] = cfg["report_title"]

    try:
        out_dir = run_alpha_selection(**kwargs)
        logger.success(f"\n🎉 {cfg['step_label']} 完成！输出目录: {out_dir}")
        return 0
    except Exception as e:
        logger.exception(f"{cfg['step_label']} 执行失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
