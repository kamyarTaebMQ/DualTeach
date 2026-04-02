"""
DualTeach Training Script
=========================
Dual-teacher knowledge distillation framework for compact language models.
Trains a 1.1B student (Gemma-3-1B) using two teachers: Gemma-2-9B (T1) and
Gemma-2-2B (T2) on the GSM8K math reasoning dataset.

Usage:
    python train.py [--lr 2e-5] [--batch_size 16] [--epochs 3] \
                   [--base_temp 4.0] [--lambda_ 1.5] [--alpha 0.7] [--beta 0.5]
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# ── GPU selection ─────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # restrict to GPU 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Model identifiers ──────────────────────────────────────────────────────────
# All Gemma-3 family: identical tokenizer (262k vocab) → no input/output ID mismatch.
# Cross-generation distillation (Gemma-2→Gemma-3) is broken because teachers
# receive Gemma-3 token IDs but interpret them as Gemma-2 tokens (completely
# different text), making both the forward pass and output logits meaningless.
MODEL_TEACHER_LARGE = "google/gemma-3-12b-it"
MODEL_TEACHER_SMALL = "google/gemma-3-4b-it"
MODEL_STUDENT       = "google/gemma-3-1b-it"


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def extract_final_number(text: str) -> str | None:
    """Extract the numeric answer after '####' in GSM8K format."""
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def format_prompt(question: str) -> str:
    return f"Problem: {question}\nAnswer:"


def load_and_preprocess_data():
    """Load GSM8K and return (train_data, test_data) as lists of dicts."""
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = list(dataset["train"])   # 7,473 samples
    test_data  = list(dataset["test"])    # 1,319 samples
    print(f"  Train: {len(train_data)} samples | Test: {len(test_data)} samples")
    return train_data, test_data


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class GSM8KDataset(Dataset):
    """
    Tokenises GSM8K samples for causal LM training.

    Labels are set to -100 for the prompt tokens (ignored in cross-entropy);
    only the answer tokens contribute to the task loss.
    """

    def __init__(self, samples, tokenizer, max_length: int = 256):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        prompt   = format_prompt(sample["question"])
        answer   = sample["answer"]   # full chain-of-thought + "#### N"
        full_text = prompt + " " + answer

        # Tokenise full text
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Mask prompt tokens in labels
        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100   # ignore prompt
        labels[attention_mask == 0] = -100  # ignore padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Vocabulary alignment  (Gemma-2 ↔ Gemma-3 have different token→ID mappings)
# ══════════════════════════════════════════════════════════════════════════════

def build_vocab_alignment(teacher_tokenizer, student_tokenizer):
    """
    Find all teacher tokens whose piece text also exists as a single token in
    the student vocabulary, building a 1-to-1 ID mapping.

    Gemma-2 (256k) and Gemma-3 (262k) share many SentencePiece pieces but assign
    them completely different integer IDs.  Distilling at the raw logit index level
    without this mapping teaches the student wrong things (e.g. the teacher's
    high-prob token for "space" lands on a completely different student token).

    Returns:
        teacher_ids: (n_aligned,) LongTensor  — teacher token IDs
        student_ids: (n_aligned,) LongTensor  — matching student token IDs
    """
    teacher_vocab = teacher_tokenizer.get_vocab()          # piece → teacher_id
    student_vocab = student_tokenizer.get_vocab()          # piece → student_id
    teacher_id_to_piece = {v: k for k, v in teacher_vocab.items()}

    t_ids, s_ids = [], []
    for t_id in sorted(teacher_id_to_piece.keys()):
        piece = teacher_id_to_piece[t_id]
        if piece in student_vocab:
            t_ids.append(t_id)
            s_ids.append(student_vocab[piece])

    pct = len(t_ids) / len(teacher_vocab) * 100
    print(f"  Vocab alignment: {len(t_ids):,}/{len(teacher_vocab):,} teacher tokens "
          f"aligned to student vocab ({pct:.1f}%)")
    return torch.tensor(t_ids, dtype=torch.long), torch.tensor(s_ids, dtype=torch.long)


def project_to_student(
    teacher_logits: torch.Tensor,   # (N, teacher_vocab) float32
    teacher_ids:    torch.Tensor,   # (n_aligned,) long  — on same device
    student_ids:    torch.Tensor,   # (n_aligned,) long  — on same device
    student_vocab:  int,
    temperature:    float,
) -> torch.Tensor:
    """
    Softmax teacher logits at `temperature`, scatter probability mass into student
    vocab space via the precomputed 1-to-1 token alignment, renormalise, and
    return log-probabilities in student vocab space.
    """
    N = teacher_logits.shape[0]
    device = teacher_logits.device

    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)   # (N, T_vocab)

    # Gather aligned probabilities
    t_ids_exp = teacher_ids.unsqueeze(0).expand(N, -1)                # (N, n_aligned)
    aligned_probs = teacher_probs.gather(1, t_ids_exp)                 # (N, n_aligned)

    # Scatter into student vocab positions
    student_probs = torch.zeros(N, student_vocab, device=device, dtype=teacher_probs.dtype)
    s_ids_exp = student_ids.unsqueeze(0).expand(N, -1)                # (N, n_aligned)
    student_probs.scatter_add_(1, s_ids_exp, aligned_probs)

    # Renormalise (unaligned tokens lose their mass; remaining mass sums to 1)
    student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (student_probs + 1e-10).log()                              # log-probs


# ══════════════════════════════════════════════════════════════════════════════
#  DualTeach core functions
# ══════════════════════════════════════════════════════════════════════════════

def combine_teachers(
    log_p1: torch.Tensor,   # (N, student_vocab) log-probs already in student space
    log_p2: torch.Tensor,   # (N, student_vocab) log-probs already in student space
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Combine two teacher log-distributions (already in student vocab space) into
    a single log-distribution via equal-weight mixture (beta=0.5).

    Uses log-sum-exp for numerical stability: log(β·p1 + (1-β)·p2).
    """
    log_beta      = torch.tensor(beta,       dtype=log_p1.dtype, device=log_p1.device).log()
    log_one_minus = torch.tensor(1.0 - beta, dtype=log_p1.dtype, device=log_p1.device).log()
    return torch.logaddexp(log_beta + log_p1, log_one_minus + log_p2)


def dualteach_loss(
    z_student: torch.Tensor,   # (batch, seq, vocab)  — all share Gemma-3 vocab
    z_t1:      torch.Tensor,   # (batch, seq, vocab)
    z_t2:      torch.Tensor,   # (batch, seq, vocab)
    labels:    torch.Tensor,   # (batch, seq)
    base_temp: float = 4.0,
    lambda_:   float = 1.5,
    alpha:     float = 0.7,
    beta:      float = 0.5,
):
    """
    Compute DualTeach total loss.

    L_total = alpha * L_distill + (1 - alpha) * L_task

    All models share the Gemma-3 tokenizer: same vocab, same token IDs on input
    and output. No alignment projection needed.

    Causal LM shift: logit[i] predicts token[i+1].
    We compare logits[:, :-1, :] to labels[:, 1:].
    Distillation is also computed over shifted answer positions to stay consistent.
    """
    vocab = z_student.size(-1)   # 262144 — teachers already trimmed to this size

    # ── Causal LM shift ────────────────────────────────────────────────────────
    # logit[i] predicts x[i+1]; compare logits[:, :-1] to labels[:, 1:]
    shift_s  = z_student[:, :-1, :].contiguous()   # (batch, seq-1, vocab)
    shift_t1 = z_t1[:, :-1, :].contiguous()
    shift_t2 = z_t2[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()       # (batch, seq-1)

    # ── Answer token mask (applied on shifted labels) ─────────────────────────
    answer_mask = (shift_labels != -100)
    mask_flat   = answer_mask.view(-1)

    if mask_flat.any():
        # Gather only answer positions — (N, vocab) float32
        t1 = shift_t1.reshape(-1, vocab)[mask_flat].float()
        t2 = shift_t2.reshape(-1, vocab)[mask_flat].float()
        s  = shift_s.reshape(-1, vocab)[mask_flat].float()

        # Adaptive temperature from KL(T1 || T2) at base temperature
        log_p1 = F.log_softmax(t1 / base_temp, dim=-1)
        log_p2 = F.log_softmax(t2 / base_temp, dim=-1)
        kl_div_value = F.kl_div(log_p2, log_p1, reduction="batchmean",
                                  log_target=True).item()
        kl_div_value = max(kl_div_value, 0.0)
        T_adaptive   = base_temp + kl_div_value * lambda_

        # Combined teacher at adaptive temperature
        log_p1_T = F.log_softmax(t1 / T_adaptive, dim=-1)
        log_p2_T = F.log_softmax(t2 / T_adaptive, dim=-1)
        z_combined   = combine_teachers(log_p1_T, log_p2_T, beta)
        student_soft = F.log_softmax(s  / T_adaptive, dim=-1)
        distill_loss = (T_adaptive ** 2) * F.kl_div(
            student_soft, z_combined, reduction="batchmean", log_target=True,
        )
    else:
        kl_div_value = 0.0
        T_adaptive   = float(base_temp)
        distill_loss = z_student.new_tensor(0.0)

    # ── Task loss (shifted cross-entropy) ─────────────────────────────────────
    task_loss = F.cross_entropy(
        shift_s.float().reshape(-1, vocab),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )

    total_loss = alpha * distill_loss + (1.0 - alpha) * task_loss
    return total_loss, distill_loss, task_loss, T_adaptive, kl_div_value


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_models(use_8bit: bool = False):
    """
    Load tokenizer + all three models onto GPU 0.

    If use_8bit=True, teachers are loaded in 8-bit via BitsAndBytesConfig.
    On H200 (143 GB VRAM) fp16 is fine; 8-bit is only needed on smaller GPUs.

    Returns:
        tokenizer, teacher_large, teacher_small, student
    """
    # bf16 has float32's dynamic range — prevents inf/NaN in Gemma-3 sliding-window attention.
    # H200 has native bf16 support; fp16 overflows on large attention logits.
    dtype    = torch.bfloat16
    gpu0_map = {"": 0}   # all layers → cuda:0

    if use_8bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            load_kw = {"quantization_config": bnb_cfg, "device_map": gpu0_map}
            print("Using 8-bit quantization for teachers (BitsAndBytesConfig) on GPU 0.")
        except Exception as e:
            print(f"WARNING: 8-bit setup failed ({e}). Falling back to bf16.")
            load_kw = {"dtype": dtype, "device_map": gpu0_map}
    else:
        load_kw = {"dtype": dtype, "device_map": gpu0_map}

    # All models share the same Gemma-3 tokenizer — no alignment needed.
    print("Loading tokenizer (shared across all Gemma-3 models)...")
    student_tokenizer = AutoTokenizer.from_pretrained(MODEL_STUDENT)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    print(f"Loading large teacher ({MODEL_TEACHER_LARGE}) → GPU 0...")
    teacher_large = AutoModelForCausalLM.from_pretrained(MODEL_TEACHER_LARGE, **load_kw)
    teacher_large.eval()
    for p in teacher_large.parameters():
        p.requires_grad = False

    print(f"Loading small teacher ({MODEL_TEACHER_SMALL}) → GPU 0...")
    teacher_small = AutoModelForCausalLM.from_pretrained(MODEL_TEACHER_SMALL, **load_kw)
    teacher_small.eval()
    for p in teacher_small.parameters():
        p.requires_grad = False

    print(f"Loading student ({MODEL_STUDENT}) → GPU 0...")
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_STUDENT,
        dtype=dtype,
        device_map=gpu0_map,
    )

    return student_tokenizer, teacher_large, teacher_small, student


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    # ── Data ──────────────────────────────────────────────────────────────────
    train_data, _ = load_and_preprocess_data()

    tokenizer, teacher_large, teacher_small, student = load_models(
        use_8bit=args.use_8bit
    )

    train_dataset = GSM8KDataset(train_data, tokenizer, max_length=args.max_length)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps   = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps  = int(0.06 * total_steps)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    device = DEVICE

    # ── Stats tracking ────────────────────────────────────────────────────────
    kl_threshold = 0.1  # threshold for "significant teacher disagreement"
    epoch_stats  = []

    print(f"\n{'='*60}")
    print(f"DualTeach Training")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size} (eff. {args.batch_size * args.grad_accum_steps})")
    print(f"  LR:               {args.lr}")
    print(f"  Base temp T:      {args.base_temp}")
    print(f"  Lambda:           {args.lambda_}")
    print(f"  Alpha:            {args.alpha}")
    print(f"  Beta:             {args.beta}")
    print(f"{'='*60}\n")

    global_step = 0

    for epoch in range(args.epochs):
        student.train()
        total_loss_epoch   = 0.0
        distill_loss_epoch = 0.0
        task_loss_epoch    = 0.0

        kl_values      = []
        high_kl_count  = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Student forward first to get reference seq_len
            out_s = student(input_ids, attention_mask=attention_mask)
            seq_len = out_s.logits.shape[1]

            # Teacher forward passes — trim vocab/seq immediately then free
            # Gemma-3 multimodal teachers have 262208-token vocab vs student 262144
            s_vocab = out_s.logits.shape[-1]
            with torch.no_grad():
                t1_logits = teacher_large(input_ids, attention_mask=attention_mask).logits
                t1_logits = t1_logits[:, :seq_len, :s_vocab].contiguous()
                torch.cuda.empty_cache()

                t2_logits = teacher_small(input_ids, attention_mask=attention_mask).logits
                t2_logits = t2_logits[:, :seq_len, :s_vocab].contiguous()
                torch.cuda.empty_cache()

            loss, d_loss, t_loss, T_adap, kl_val = dualteach_loss(
                out_s.logits,
                t1_logits,
                t2_logits,
                labels,
                base_temp=args.base_temp,
                lambda_=args.lambda_,
                alpha=args.alpha,
                beta=args.beta,
            )

            # Track disagreement stats
            kl_values.append(kl_val)
            if kl_val > kl_threshold:
                high_kl_count += 1

            # Gradient accumulation
            (loss / args.grad_accum_steps).backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss_epoch   += loss.item()
            distill_loss_epoch += d_loss.item()
            task_loss_epoch    += t_loss.item()

            if step % 100 == 0:
                pbar.set_postfix({
                    "loss":    f"{loss.item():.4f}",
                    "distill": f"{d_loss.item():.4f}",
                    "task":    f"{t_loss.item():.4f}",
                    "T_adap":  f"{T_adap:.2f}",
                    "kl":      f"{kl_val:.4f}",
                })

        n_batches   = len(train_loader)
        mean_kl     = float(np.mean(kl_values))
        pct_high_kl = high_kl_count / n_batches * 100.0

        stats = {
            "epoch":          epoch + 1,
            "avg_loss":       round(total_loss_epoch   / n_batches, 4),
            "avg_distill":    round(distill_loss_epoch / n_batches, 4),
            "avg_task":       round(task_loss_epoch    / n_batches, 4),
            "mean_kl":        round(mean_kl, 4),
            "pct_high_kl":    round(pct_high_kl, 1),
        }
        epoch_stats.append(stats)

        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Avg loss:        {stats['avg_loss']}")
        print(f"  Avg distill:     {stats['avg_distill']}")
        print(f"  Avg task:        {stats['avg_task']}")
        print(f"  Mean KL (T1||T2): {stats['mean_kl']:.4f}")
        print(f"  Batches KL>{kl_threshold}: {stats['pct_high_kl']:.1f}%  "
              f"(paper reports ~53% of batches show notable disagreement)\n")

        # Save checkpoint
        ckpt_dir = Path(args.output_dir) / f"epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        print(f"Checkpoint saved → {ckpt_dir}")

    # Save training stats
    stats_path = Path("results") / "training_stats.json"
    stats_path.parent.mkdir(exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"hyperparams": vars(args), "epoch_stats": epoch_stats}, f, indent=2)
    print(f"\nTraining stats saved → {stats_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="DualTeach: Dual-Teacher Knowledge Distillation for GSM8K"
    )

    # Paths
    p.add_argument("--output_dir",  default="checkpoints/student",
                   help="Directory to save student checkpoints.")

    # Hyperparams — optimal values from paper ablations as defaults
    p.add_argument("--lr",           type=float, default=2e-5,
                   help="AdamW learning rate (default: 2e-5).")
    p.add_argument("--batch_size",   type=int,   default=4,
                   help="Per-device batch size (default: 4 with 12B teacher).")
    p.add_argument("--grad_accum_steps", type=int, default=16,
                   help="Gradient accumulation steps → effective batch = batch_size * grad_accum_steps.")
    p.add_argument("--epochs",       type=int,   default=3,
                   help="Number of training epochs (default: 3).")
    p.add_argument("--max_length",   type=int,   default=256,
                   help="Max token sequence length (default: 256).")
    p.add_argument("--num_workers",  type=int,   default=4,
                   help="DataLoader worker processes.")

    # DualTeach hyperparams
    p.add_argument("--base_temp",  type=float, default=4.0,
                   help="Base distillation temperature T (default: 4.0 — ablation optimal).")
    p.add_argument("--lambda_",    type=float, default=1.5,
                   help="Adaptive temperature strength λ (default: 1.5 — ablation optimal).")
    p.add_argument("--alpha",      type=float, default=0.7,
                   help="Distillation loss weight α (default: 0.7 — ablation optimal).")
    p.add_argument("--beta",       type=float, default=0.5,
                   help="Teacher combination weight β (default: 0.5 — equal weighting).")

    # Memory options
    p.add_argument("--use_8bit", action="store_true",
                   help="Load teachers in 8-bit precision via bitsandbytes (saves ~50%% VRAM).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
