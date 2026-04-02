# DualTeach: Dual-Teacher Knowledge Distillation for Compact Math Reasoning Models

DualTeach is a knowledge distillation framework that trains a **1.1B student model** (Gemma-3-1B-IT) using **two teachers simultaneously** — a large teacher (Gemma-3-12B-IT) and a small teacher (Gemma-3-4B-IT) — with an **adaptive temperature** mechanism that dynamically adjusts distillation difficulty based on teacher disagreement.

---

## Key Idea

Standard single-teacher KD uses a fixed temperature. DualTeach computes an adaptive temperature at each training step:

```
T_adaptive = T_base + λ × KL(T1 || T2)
```

When the two teachers disagree (high KL divergence), the temperature rises — softening the target distribution and giving the student more room to learn. When they agree, the temperature drops — sharpening the signal.

The final distillation loss blends both teachers:

```
L_distill = β · KL(student || T1) + (1 − β) · KL(student || T2)
L_total   = α · L_distill + (1 − α) · L_task
```

---

## Models

| Role | Model | VRAM |
|---|---|---|
| Student | google/gemma-3-1b-it | 2.0 GB |
| Teacher T1 (large) | google/gemma-3-12b-it | 24.38 GB |
| Teacher T2 (small) | google/gemma-3-4b-it | ~8 GB |
| All together (training) | — | 34.97 GB |

---

## Results

### GSM8K (trained in-distribution)

| Model | GSM8K |
|---|---|
| Gemma-3-1B-IT (no fine-tuning) | 8.0% |
| Single-T2 (4B teacher only) | 24.9% |
| Single-T1 (12B teacher only) | 25.9% |
| **DualTeach (ours)** | **30.5%** |

DualTeach outperforms both single-teacher baselines by **4.6–5.6 percentage points**.

### AQuA-RAT (multiple-choice math, trained in-distribution)

| Model | AQuA-RAT |
|---|---|
| Gemma-3-1B-IT (no fine-tuning) | 17.3% |
| Single-T2 (AQuA-trained) | 21.3% |
| Single-T1 (AQuA-trained) | 23.2% |
| **DualTeach α=0.3 (ours)** | **25.2%** |

### Deployment Efficiency

| | Student (1.1B) | Teacher T1 (12B) |
|---|---|---|
| VRAM | **2.0 GB** | 24.38 GB |
| Latency (p50, concurrency=1) | **1,556 ms** | 2,890 ms |
| Throughput | **0.65 req/s** | 0.35 req/s |
| Success rate @ concurrency=50 | **100%** | 82% |

The distilled student is **1.86× faster**, uses **12× less VRAM**, and is more reliable under load — enabling deployment on consumer GPUs (≥4 GB VRAM).

---

## Installation

```bash
git clone https://github.com/your-username/DualTeach
cd DualTeach
pip install -r requirements.txt
```

Requires Python 3.10+ and CUDA-capable GPU. Training all three models simultaneously requires ~35 GB VRAM (e.g. H100/H200). Inference requires only 2 GB.

---

## Training

### GSM8K

```bash
# DualTeach (dual-teacher, adaptive temperature)
python train.py --lambda_ 1.5 --alpha 0.7 --beta 0.5

# Single-teacher ablations
python train.py --beta 1.0 --lambda_ 0   # Single-T1 only
python train.py --beta 0.0 --lambda_ 0   # Single-T2 only
```

### AQuA-RAT

```bash
python train_aqua.py --lambda_ 1.5 --alpha 0.3 --beta 0.5
```

### MATH (Henderson et al.)

```bash
python train_math.py --lambda_ 1.5 --alpha 0.7 --beta 0.5
```

**Key hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `--lambda_` | 1.5 | Adaptive temperature scaling factor |
| `--alpha` | 0.7 | Weight on distillation loss (vs task loss) |
| `--beta` | 0.5 | Weight on T1 (vs T2) in distillation |
| `--base_temp` | 4.0 | Base distillation temperature |
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 2e-5 | Learning rate |

Checkpoints are saved to `checkpoints/<run_name>/epoch_<n>/`.

---

## Evaluation

### GSM8K

```bash
python evaluate.py
```

### MATH and AQuA-RAT

```bash
# Single model
python evaluate_new_benchmarks.py --model dualteach --benchmark math
python evaluate_new_benchmarks.py --model dualteach_aqua_a03 --benchmark aqua

# All models
python evaluate_new_benchmarks.py
```

Available model keys: `dualteach`, `single_t1`, `single_t2`, `baseline`, `math_dualteach`, `math_single_t1`, `math_single_t2`, `dualteach_aqua`, `dualteach_aqua_a03`, `aqua_single_t1`, `aqua_single_t2`, `base`.

---

## FastAPI Service & Benchmark

Serve the student model:

```bash
python serve.py --model checkpoints/student/epoch_5
```

Run the concurrency benchmark:

```bash
python run_benchmark.py
```

Results are saved to `results/benchmark_results.json`.

---

## Repository Structure

```
DualTeach/
├── train.py                    # DualTeach training (GSM8K)
├── train_aqua.py               # DualTeach training (AQuA-RAT)
├── train_math.py               # DualTeach training (MATH)
├── evaluate.py                 # GSM8K evaluation
├── evaluate_new_benchmarks.py  # MATH + AQuA-RAT evaluation
├── finetune_baseline.py        # Standard fine-tuning baseline
├── ablations.py                # Single-teacher ablation runs
├── serve.py                    # FastAPI inference server
├── run_benchmark.py            # Concurrency benchmark
├── print_results_table.py      # Results table printer
└── requirements.txt
```

---

## Citation

If you use DualTeach in your research, please cite:

```bibtex
@article{dualteach2026,
  title   = {DualTeach: Adaptive Dual-Teacher Knowledge Distillation for Compact Math Reasoning},
  author  = {Your Name},
  year    = {2026},
}
```

---

## License

MIT License
