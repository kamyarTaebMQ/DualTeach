"""
DualTeach Ablation Study Runner
=================================
Runs the four ablation sets described in the paper.

Ablation A — Base Temperature T ∈ {2, 3, 4, 6}
    Fix α=0.7, λ=1.5.  Train 3 epochs each.

Ablation B — Loss Weight α ∈ {0.5, 0.6, 0.7, 0.8}
    Fix T=4, λ=1.5.

Ablation C — Fixed vs Adaptive Temperature
    DualTeach fixed T (β=0.5, λ=0) vs DualTeach adaptive (β=0.5, λ=1.5).

Ablation D — Single vs Dual Teacher
    Baseline (no distillation) / Single-T2 / Single-T1 / DualTeach.

Each ablation trains a separate student model and evaluates on GSM8K test.
Results are saved to results/ablation_results.json.

Usage:
    # Run all ablations
    python ablations.py --all

    # Run specific ablation
    python ablations.py --ablation A
    python ablations.py --ablation B
    python ablations.py --ablation C
    python ablations.py --ablation D

    # Dry run (print configs without training)
    python ablations.py --all --dry_run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  Ablation configurations
# ══════════════════════════════════════════════════════════════════════════════

# Ablation A — Temperature
ABLATION_A = [
    {"name": "T=2", "base_temp": 2.0, "lambda_": 1.5, "alpha": 0.7},
    {"name": "T=3", "base_temp": 3.0, "lambda_": 1.5, "alpha": 0.7},
    {"name": "T=4", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.7},  # optimal
    {"name": "T=6", "base_temp": 6.0, "lambda_": 1.5, "alpha": 0.7},
]

# Ablation B — Loss weight alpha
ABLATION_B = [
    {"name": "alpha=0.5", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.5},
    {"name": "alpha=0.6", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.6},
    {"name": "alpha=0.7", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.7},  # optimal
    {"name": "alpha=0.8", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.8},
]

# Ablation C — Fixed vs adaptive temperature
ABLATION_C = [
    {"name": "fixed_T",    "base_temp": 4.0, "lambda_": 0.0, "alpha": 0.7},  # λ=0 → no adaptation
    {"name": "adaptive_T", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.7},  # main result
]

# Ablation D — Teacher configuration
# baseline: alpha=0 (no distillation, pure task loss)
# single_T2: beta=1.0 → combine_teachers returns log_p2 only
# single_T1: beta=0.0 → combine_teachers returns log_p1 only
# DualTeach: beta=0.5 (equal mixture of both teachers)
ABLATION_D = [
    {"name": "baseline",  "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.0, "beta": 0.5},
    {"name": "single_T2", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.7, "beta": 1.0},
    {"name": "single_T1", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.7, "beta": 0.0},
    {"name": "DualTeach", "base_temp": 4.0, "lambda_": 1.5, "alpha": 0.7, "beta": 0.5},
]

# Expected results from paper (for sanity checking)
EXPECTED_RESULTS = {
    "baseline":   0.420,
    "single_T2":  0.452,
    "single_T1":  0.475,
    "fixed_T":    0.510,
    "DualTeach":  0.522,
    "T=4":        0.522,
    "alpha=0.7":  0.522,
    "adaptive_T": 0.522,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Training helper
# ══════════════════════════════════════════════════════════════════════════════

def run_training(config: dict, output_dir: str, dry_run: bool = False) -> str:
    """
    Launch train.py with the given hyperparameter config as a subprocess.

    Args:
        config:     dict of hyperparams (name, base_temp, lambda_, alpha, ...)
        output_dir: where to save the checkpoint
        dry_run:    if True, print the command without running it

    Returns:
        path to the final epoch checkpoint
    """
    cmd = [
        sys.executable, "train.py",
        "--output_dir",       output_dir,
        "--base_temp",        str(config.get("base_temp", 4.0)),
        "--lambda_",          str(config.get("lambda_", 1.5)),
        "--alpha",            str(config.get("alpha", 0.7)),
        "--beta",             str(config.get("beta", 0.5)),
        "--epochs",           "3",
        "--batch_size",       "2",
        "--grad_accum_steps", "32",
    ]

    print(f"\n{'─'*60}")
    print(f"Training config: {config['name']}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN] Skipping actual training.")
        return f"{output_dir}/epoch_3"

    result = subprocess.run(cmd, check=True)
    return f"{output_dir}/epoch_3"


def run_evaluation(checkpoint: str, dry_run: bool = False) -> float:
    """
    Run evaluate.py on a checkpoint and return accuracy.

    Returns:
        accuracy as float in [0, 1]
    """
    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", checkpoint,
        "--output",     f"{checkpoint}/eval_results.json",
    ]

    print(f"  Evaluating: {checkpoint}")

    if dry_run:
        import random
        fake_acc = 0.42 + random.uniform(-0.02, 0.10)
        print(f"  [DRY RUN] Fake accuracy: {fake_acc:.4f}")
        return fake_acc

    subprocess.run(cmd, check=True)

    result_file = Path(checkpoint) / "eval_results.json"
    with open(result_file) as f:
        data = json.load(f)

    acc = data.get(checkpoint, {}).get("accuracy", 0.0)
    return acc


# ══════════════════════════════════════════════════════════════════════════════
#  Ablation runners
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(label: str, configs: list[dict], dry_run: bool = False) -> list[dict]:
    """
    Run a full ablation set: train + eval for each config.

    Returns:
        list of result dicts: {"name": ..., "accuracy": ..., "delta_pp": ...}
    """
    print(f"\n{'='*60}")
    print(f"ABLATION {label}")
    print(f"{'='*60}")

    results = []

    for config in configs:
        name       = config["name"]
        output_dir = f"checkpoints/ablation_{label.lower()}_{name.replace('=', '').replace('.', '')}"

        try:
            ckpt_dir = run_training(config, output_dir, dry_run)
            accuracy = run_evaluation(ckpt_dir, dry_run)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR during {name}: {e}")
            accuracy = None

        expected = EXPECTED_RESULTS.get(name, None)
        delta    = (accuracy - 0.42) * 100 if accuracy is not None else None

        result = {
            "name":       name,
            "config":     config,
            "accuracy":   round(accuracy, 4) if accuracy is not None else None,
            "accuracy_pct": round(accuracy * 100, 1) if accuracy is not None else None,
            "delta_pp":   round(delta, 1) if delta is not None else None,
            "expected":   expected,
        }

        if accuracy is not None and expected is not None:
            diff = abs(accuracy - expected)
            result["matches_paper"] = diff < 0.02  # within 2pp tolerance

        results.append(result)
        print(f"  {name}: {accuracy*100:.1f}%" if accuracy else f"  {name}: FAILED")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def print_ablation_table(label: str, results: list[dict]):
    """Print a formatted ablation results table."""
    print(f"\n{'─'*50}")
    print(f"Ablation {label} Results:")
    print(f"{'─'*50}")
    print(f"{'Config':<20} {'Accuracy':>10} {'ΔGSM8K':>10} {'Expected':>10}")
    print(f"{'─'*50}")
    for r in results:
        acc  = f"{r['accuracy_pct']:.1f}%" if r['accuracy_pct'] else "FAILED"
        delta = f"+{r['delta_pp']:.1f}pp" if r['delta_pp'] else "—"
        exp  = f"{r['expected']*100:.1f}%" if r['expected'] else "—"
        print(f"{r['name']:<20} {acc:>10} {delta:>10} {exp:>10}")
    print(f"{'─'*50}")


def save_all_results(all_results: dict):
    """Save aggregated ablation results to JSON."""
    out = Path("results/ablation_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAblation results saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DualTeach Ablation Study Runner")
    p.add_argument("--ablation", choices=["A", "B", "C", "D"],
                   help="Run a specific ablation (A/B/C/D).")
    p.add_argument("--all", action="store_true",
                   help="Run all ablations sequentially.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print configs and commands without running training.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.all and not args.ablation:
        print("Specify --all or --ablation [A|B|C|D].")
        return

    all_results = {}

    ablations_to_run = (
        ["A", "B", "C", "D"] if args.all
        else [args.ablation]
    )

    ablation_map = {
        "A": ("Temperature T",         ABLATION_A),
        "B": ("Loss Weight α",         ABLATION_B),
        "C": ("Fixed vs Adaptive T",   ABLATION_C),
        "D": ("Single vs Dual Teacher",ABLATION_D),
    }

    for key in ablations_to_run:
        label, configs = ablation_map[key]
        results = run_ablation(key, configs, dry_run=args.dry_run)
        print_ablation_table(key, results)
        all_results[f"ablation_{key}"] = {
            "description": label,
            "results":     results,
        }

    save_all_results(all_results)

    # Print summary
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    for key, data in all_results.items():
        print(f"\n{data['description']}:")
        for r in data["results"]:
            match = " ✓" if r.get("matches_paper") else " (differs from paper)"
            acc   = f"{r['accuracy_pct']:.1f}%" if r['accuracy_pct'] else "FAILED"
            print(f"  {r['name']:<20}: {acc}{match}")


if __name__ == "__main__":
    main()
