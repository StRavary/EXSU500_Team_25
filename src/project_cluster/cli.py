"""
cli.py
Unified command-line interface for EXSU500_Team_25 cardiometabolic phenotyping.
"""

import argparse
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _run_stage(script_name: str) -> None:
    script_path = Path(__file__).parent / script_name
    spec = spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cardiometabolic-pheno",
        description="NHANES Cardiometabolic Phenotyping & Disparity Auditing Pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "prepare", help="Filter N=3,515 cohort, merge CDC weights, and split 70/30"
    )
    sub.add_parser(
        "tune", help="Run 5-fold CV tuning across k=2..6 for k-Means, GMM, and AE"
    )
    sub.add_parser(
        "stability", help="Run 100-bootstrap Jaccard and 100-permutation null audits"
    )
    sub.add_parser(
        "evaluate", help="Run locked held-out test evaluation and FDR disparity tests"
    )
    sub.add_parser("all", help="Execute the full 4-stage pipeline sequentially")
    sub.add_parser(
        "ui", help="Launch the interactive Streamlit clinical phenotyping dashboard"
    )

    args = parser.parse_args()

    if args.command == "prepare":
        _run_stage("01_prepare_cohort.py")
    elif args.command == "tune":
        _run_stage("02_train_and_tune.py")
    elif args.command == "stability":
        _run_stage("03_stability_and_null.py")
    elif args.command == "evaluate":
        _run_stage("04_evaluate_test_and_disparities.py")
    elif args.command == "all":
        for stage in [
            "01_prepare_cohort.py",
            "02_train_and_tune.py",
            "03_stability_and_null.py",
            "04_evaluate_test_and_disparities.py",
        ]:
            _run_stage(stage)
    elif args.command == "ui":
        app_path = Path(__file__).parent / "app.py"
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)], check=True
        )


if __name__ == "__main__":
    main()
