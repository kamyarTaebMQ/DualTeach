import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class AdaptiveWeightingModule(nn.Module):
    """Learnable adaptive weighting for dual teachers"""
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, teacher1_logits: torch.Tensor, teacher2_logits: torch.Tensor) -> torch.Tensor:
        # Compute features from logits
        t1_features = teacher1_logits.mean(dim=1)  # [batch_size, vocab_size] -> [batch_size, vocab_size]
        t2_features = teacher2_logits.mean(dim=1)
        
        # Concatenate and compute weights
        combined = torch.cat([t1_features, t2_features], dim=-1)
        weights = self.weight_net(combined)  # [batch_size, 2]
        
        return weights

class DualTeacherDistillationTrainer(Trainer):
    def __init__(
        self,
        teacher1_model,
        teacher2_model, 
        student_model,
        teacher1_tokenizer,
        teacher2_tokenizer,
        student_tokenizer,
        temperature: float = 4.0,
        alpha: float = 0.7,
        **kwargs
    ):
        super().__init__(model=student_model, **kwargs)
        
        self.teacher1 = teacher1_model.eval()
        self.teacher2 = teacher2_model.eval()
        self.teacher1_tokenizer = teacher1_tokenizer
        self.teacher2_tokenizer = teacher2_tokenizer
        self.student_tokenizer = student_tokenizer
        
        self.temperature = temperature
        self.alpha = alpha
        
        # Adaptive weighting module
        self.adaptive_weights = AdaptiveWeightingModule().to(self.model.device)
        
        # Freeze teachers
        for param in self.teacher1.parameters():
            param.requires_grad = False
        for param in self.teacher2.parameters():
            param.requires_grad = False
            
        # Add adaptive weights to optimizer
        self.adaptive_optimizer = torch.optim.AdamW(
            self.adaptive_weights.parameters(), 
            lr=1e-4
        )
        
    def align_vocabularies(self, teacher_logits: torch.Tensor, teacher_tokenizer, student_tokenizer) -> torch.Tensor:
        """Simplified vocabulary alignment"""
        # For now, just truncate to smaller vocabulary size to avoid indexing errors
        batch_size, seq_len, teacher_vocab_size = teacher_logits.shape
        student_vocab_size = len(student_tokenizer)
        
        min_vocab_size = min(teacher_vocab_size, student_vocab_size)
        
        # Truncate to common vocabulary size
        if teacher_vocab_size > min_vocab_size:
            return teacher_logits[:, :, :min_vocab_size]
        elif student_vocab_size > min_vocab_size:
            # Pad teacher logits to match student vocab size
            padding = torch.full(
                (batch_size, seq_len, student_vocab_size - teacher_vocab_size),
                -float('inf'),
                device=teacher_logits.device,
                dtype=teacher_logits.dtype
            )
            return torch.cat([teacher_logits, padding], dim=-1)
        else:
            return teacher_logits
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute dual-teacher distillation loss"""
        labels = inputs.get("labels")
        
        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Get teacher predictions
        with torch.no_grad():
            # Teacher 1 (Gemma 2-9B)
            teacher1_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
            teacher1_outputs = self.teacher1(**teacher1_inputs)
            teacher1_logits = self.align_vocabularies(
                teacher1_outputs.logits, 
                self.teacher1_tokenizer, 
                self.student_tokenizer
            )
            
            # Teacher 2 (Qwen 2.5-7B)
            teacher2_outputs = self.teacher2(**teacher1_inputs)  # Same input format
            teacher2_logits = self.align_vocabularies(
                teacher2_outputs.logits,
                self.teacher2_tokenizer,
                self.student_tokenizer
            )
        
        # Compute adaptive weights
        weights = self.adaptive_weights(
            teacher1_logits.detach(), 
            teacher2_logits.detach()
        )  # [batch_size, 2]
        
        # Weighted combination of teacher logits
        w1 = weights[:, 0:1].unsqueeze(-1)  # [batch_size, 1, 1]
        w2 = weights[:, 1:2].unsqueeze(-1)  # [batch_size, 1, 1]
        
        combined_teacher_logits = w1 * teacher1_logits + w2 * teacher2_logits
        
        # Distillation loss
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(combined_teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Task loss (if labels provided)
        task_loss = 0
        if labels is not None:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        
        # Update adaptive weights
        self.adaptive_optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.adaptive_optimizer.step()
        
        return (total_loss, student_outputs) if return_outputs else total_loss

def setup_dual_teacher_training(
    teacher1_model_name: str = "google/gemma-2-9b-it",
    teacher2_model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
    student_model_name: str = "google/gemma-3-1b-it",
    output_dir: str = "./dual_teacher_results",
    num_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 5e-5
):
    """Setup dual-teacher distillation training"""
    
    # Load models and tokenizers
    print("Loading teacher models...")
    teacher1_tokenizer = AutoTokenizer.from_pretrained(teacher1_model_name)
    teacher1_model = AutoModelForCausalLM.from_pretrained(
        teacher1_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    teacher2_tokenizer = AutoTokenizer.from_pretrained(teacher2_model_name)
    teacher2_model = AutoModelForCausalLM.from_pretrained(
        teacher2_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading student model...")
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Add padding tokens if missing
    for tokenizer in [teacher1_tokenizer, teacher2_tokenizer, student_tokenizer]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        save_strategy="epoch",
        eval_strategy="no",
        logging_steps=10,
        remove_unused_columns=False,
        report_to=None
    )
    
    return (
        teacher1_model, teacher2_model, student_model,
        teacher1_tokenizer, teacher2_tokenizer, student_tokenizer,
        training_args
    )

# Example usage
if __name__ == "__main__":
    # Setup models
    models_and_tokenizers = setup_dual_teacher_training()
    teacher1_model, teacher2_model, student_model = models_and_tokenizers[:3]
    teacher1_tokenizer, teacher2_tokenizer, student_tokenizer = models_and_tokenizers[3:6]
    training_args = models_and_tokenizers[6]
    
    # Create trainer (you'll need to add your dataset)
    trainer = DualTeacherDistillationTrainer(
        teacher1_model=teacher1_model,
        teacher2_model=teacher2_model,
        student_model=student_model,
        teacher1_tokenizer=teacher1_tokenizer,
        teacher2_tokenizer=teacher2_tokenizer,
        student_tokenizer=student_tokenizer,
        args=training_args,
        # train_dataset=your_dataset,  # Add your GSM8K dataset here
        temperature=4.0,
        alpha=0.7
    )