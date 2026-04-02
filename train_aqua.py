"""
train_aqua.py
=============
DualTeach training on AQuA-RAT (multiple-choice math reasoning).

Identical to train.py except the data layer:
  - Dataset: AQuA-RAT (aqua_rat, 97k train samples; we use a 20k subset)
  - Prompt:  question + lettered options (MC format)
  - Answer:  rationale + "The answer is X." (chain-of-thought + final letter)

All DualTeach machinery (adaptive temperature, dual-teacher KD loss,
vocab alignment) is unchanged.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_aqua.py \
        --output_dir checkpoints/aqua_dualteach \
        --epochs 3 --batch_size 2 --grad_accum_steps 32
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# ── GPU ───────────────────────────────────────────────────────────────────────
# Honour CUDA_VISIBLE_DEVICES if set externally; default to GPU 0
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Model identifiers (same Gemma-3 family as train.py) ───────────────────────
MODEL_TEACHER_LARGE = "google/gemma-3-12b-it"
MODEL_TEACHER_SMALL = "google/gemma-3-4b-it"
MODEL_STUDENT       = "google/gemma-3-1b-it"


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers — AQuA-RAT specific
# ══════════════════════════════════════════════════════════════════════════════

def format_aqua_prompt(question: str, options: list) -> str:
    """
    Format AQuA-RAT question + options as an MC prompt.
    Options arrive as e.g. ["A)21", "B)21.5", ...]; already letter-prefixed.
    Consistent with make_aqua_prompt() in evaluate_new_benchmarks.py.
    """
    letters = "ABCDE"
    lines = []
    for i, opt in enumerate(options):
        opt = opt.strip()
        if opt and opt[0].upper() in letters and len(opt) > 1 and opt[1] == ")":
            lines.append(opt)
        else:
            lines.append(f"{letters[i]}) {opt}")
    return (
        "Solve the following multiple-choice math problem. "
        "Think step by step, then give your final answer as a single letter (A, B, C, D, or E).\n\n"
        f"Question: {question}\n"
        "Options:\n" + "\n".join(lines) + "\n"
        "Answer:"
    )


def format_aqua_answer(rationale: str, correct: str) -> str:
    """
    Format target answer: rationale (chain-of-thought) + final letter.
    Using the rationale gives richer supervision than just the letter.
    Example output: "If Q completes x km... The answer is E."
    """
    r = rationale.strip()
    # Append explicit letter at the end if not already there
    if not r.endswith(f"The answer is {correct}."):
        r = r + f"\nThe answer is {correct}."
    return r


def load_aqua_data(n_train: int = 20000):
    """
    Load AQuA-RAT and return a random subset of training samples.

    AQuA-RAT fields:
      question (str), options (list[str]), rationale (str), correct (str)

    We randomly sample n_train examples from 97k for tractable training time.
    """
    print(f"Loading AQuA-RAT (sampling {n_train} from training split)...")
    ds = load_dataset("aqua_rat", "raw")
    train_samples = list(ds["train"])
    random.shuffle(train_samples)
    subset = train_samples[:n_train]
    print(f"  Selected {len(subset)} training samples.")
    return subset


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class AQuaDataset(Dataset):
    """
    Tokenises AQuA-RAT samples for causal LM training.

    Labels are -100 for prompt tokens (not in loss); answer tokens contribute
    to both task loss and distillation (same masking as GSM8KDataset).
    """

    def __init__(self, samples, tokenizer, max_length: int = 512):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt    = format_aqua_prompt(s["question"], s["options"])
        answer    = format_aqua_answer(s["rationale"], s["correct"])
        full_text = prompt + " " + answer

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Mask prompt tokens in labels so loss is only on the answer
        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len]             = -100  # ignore prompt
        labels[attention_mask == 0]     = -100  # ignore padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  DualTeach loss  (identical to train.py)
# ══════════════════════════════════════════════════════════════════════════════

def combine_teachers(log_p1, log_p2, beta=0.5):
    log_beta      = torch.tensor(beta,       dtype=log_p1.dtype, device=log_p1.device).log()
    log_one_minus = torch.tensor(1.0 - beta, dtype=log_p1.dtype, device=log_p1.device).log()
    return torch.logaddexp(log_beta + log_p1, log_one_minus + log_p2)


def dualteach_loss(z_student, z_t1, z_t2, labels,
                   base_temp=4.0, lambda_=1.5, alpha=0.7, beta=0.5):
    vocab = z_student.size(-1)

    shift_s      = z_student[:, :-1, :].contiguous()
    shift_t1     = z_t1[:, :-1, :].contiguous()
    shift_t2     = z_t2[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    answer_mask = (shift_labels != -100)
    mask_flat   = answer_mask.view(-1)

    if mask_flat.any():
        t1 = shift_t1.reshape(-1, vocab)[mask_flat].float()
        t2 = shift_t2.reshape(-1, vocab)[mask_flat].float()
        s  = shift_s.reshape(-1, vocab)[mask_flat].float()

        log_p1 = F.log_softmax(t1 / base_temp, dim=-1)
        log_p2 = F.log_softmax(t2 / base_temp, dim=-1)
        kl_div_value = F.kl_div(log_p2, log_p1, reduction="batchmean",
                                  log_target=True).item()
        kl_div_value = max(kl_div_value, 0.0)
        T_adaptive   = base_temp + kl_div_value * lambda_

        log_p1_T     = F.log_softmax(t1 / T_adaptive, dim=-1)
        log_p2_T     = F.log_softmax(t2 / T_adaptive, dim=-1)
        z_combined   = combine_teachers(log_p1_T, log_p2_T, beta)
        student_soft = F.log_softmax(s / T_adaptive, dim=-1)
        distill_loss = (T_adaptive ** 2) * F.kl_div(
            student_soft, z_combined, reduction="batchmean", log_target=True,
        )
    else:
        kl_div_value = 0.0
        T_adaptive   = float(base_temp)
        distill_loss = z_student.new_tensor(0.0)

    task_loss = F.cross_entropy(
        shift_s.float().reshape(-1, vocab),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    total_loss = alpha * distill_loss + (1.0 - alpha) * task_loss
    return total_loss, distill_loss, task_loss, T_adaptive, kl_div_value


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading  (identical to train.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    dtype    = torch.bfloat16
    gpu0_map = {"": 0}

    print("Loading shared Gemma-3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_STUDENT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading T1 ({MODEL_TEACHER_LARGE}) → GPU 0...")
    t1 = AutoModelForCausalLM.from_pretrained(MODEL_TEACHER_LARGE,
                                               torch_dtype=dtype, device_map=gpu0_map)
    t1.eval()
    for p in t1.parameters():
        p.requires_grad = False

    print(f"Loading T2 ({MODEL_TEACHER_SMALL}) → GPU 0...")
    t2 = AutoModelForCausalLM.from_pretrained(MODEL_TEACHER_SMALL,
                                               torch_dtype=dtype, device_map=gpu0_map)
    t2.eval()
    for p in t2.parameters():
        p.requires_grad = False

    print(f"Loading student ({MODEL_STUDENT}) → GPU 0...")
    student = AutoModelForCausalLM.from_pretrained(MODEL_STUDENT,
                                                    torch_dtype=dtype, device_map=gpu0_map)

    vram = round(torch.cuda.memory_allocated() / 1e9, 2)
    print(f"All models loaded. VRAM: {vram} GB")
    return tokenizer, t1, t2, student


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    train_data = load_aqua_data(n_train=args.n_train)
    tokenizer, teacher_large, teacher_small, student = load_models()

    dataset = AQuaDataset(train_data, tokenizer, max_length=args.max_length)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=True)

    optimizer    = AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = (len(loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    kl_threshold = 0.1
    epoch_stats  = []

    print(f"\n{'='*60}")
    print(f"DualTeach Training — AQuA-RAT")
    print(f"  Train samples:    {len(train_data)}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size} (eff. {args.batch_size * args.grad_accum_steps})")
    print(f"  LR:               {args.lr}")
    print(f"  Base temp:        {args.base_temp}")
    print(f"  Lambda:           {args.lambda_}")
    print(f"  Alpha:            {args.alpha}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        student.train()
        total_loss = distill_loss_sum = task_loss_sum = 0.0
        kl_values = []
        high_kl   = 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            out_s   = student(input_ids, attention_mask=attention_mask)
            seq_len = out_s.logits.shape[1]
            s_vocab = out_s.logits.shape[-1]

            with torch.no_grad():
                t1_logits = teacher_large(input_ids, attention_mask=attention_mask).logits
                t1_logits = t1_logits[:, :seq_len, :s_vocab].contiguous()
                torch.cuda.empty_cache()

                t2_logits = teacher_small(input_ids, attention_mask=attention_mask).logits
                t2_logits = t2_logits[:, :seq_len, :s_vocab].contiguous()
                torch.cuda.empty_cache()

            loss, d_loss, t_loss, T_adap, kl_val = dualteach_loss(
                out_s.logits, t1_logits, t2_logits, labels,
                base_temp=args.base_temp, lambda_=args.lambda_,
                alpha=args.alpha, beta=args.beta,
            )

            kl_values.append(kl_val)
            if kl_val > kl_threshold:
                high_kl += 1

            (loss / args.grad_accum_steps).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss     += loss.item()
            distill_loss_sum += d_loss.item()
            task_loss_sum    += t_loss.item()

            if step % 100 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "distill": f"{d_loss.item():.4f}",
                    "task": f"{t_loss.item():.4f}",
                    "T": f"{T_adap:.2f}",
                    "kl": f"{kl_val:.4f}",
                })

        n = len(loader)
        stats = {
            "epoch":       epoch + 1,
            "avg_loss":    round(total_loss / n, 4),
            "avg_distill": round(distill_loss_sum / n, 4),
            "avg_task":    round(task_loss_sum / n, 4),
            "mean_kl":     round(float(np.mean(kl_values)), 4),
            "pct_high_kl": round(high_kl / n * 100, 1),
        }
        epoch_stats.append(stats)
        print(f"\nEpoch {epoch+1} complete: loss={stats['avg_loss']}  "
              f"distill={stats['avg_distill']}  task={stats['avg_task']}  "
              f"kl={stats['mean_kl']}")

        ckpt_dir = Path(args.output_dir) / f"epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        print(f"Checkpoint saved → {ckpt_dir}")

    stats_path = Path("results") / "aqua_training_stats.json"
    stats_path.parent.mkdir(exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"hyperparams": vars(args), "epoch_stats": epoch_stats}, f, indent=2)
    print(f"\nTraining stats saved → {stats_path}")
    print("ALL_DONE")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DualTeach on AQuA-RAT")
    p.add_argument("--output_dir",       default="checkpoints/aqua_dualteach")
    p.add_argument("--n_train",          type=int,   default=20000,
                   help="Number of training samples to use (default: 20000)")
    p.add_argument("--lr",               type=float, default=2e-5)
    p.add_argument("--batch_size",       type=int,   default=2)
    p.add_argument("--grad_accum_steps", type=int,   default=32)
    p.add_argument("--epochs",           type=int,   default=3)
    p.add_argument("--max_length",       type=int,   default=512)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--base_temp",        type=float, default=4.0)
    p.add_argument("--lambda_",          type=float, default=1.5)
    p.add_argument("--alpha",            type=float, default=0.7)
    p.add_argument("--beta",             type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
