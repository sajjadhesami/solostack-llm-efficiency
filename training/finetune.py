"""
Lightweight QLoRA fine-tuning script with subject-specific dataset support.

Usage (pre-download models and data before running offline):
  # Train 4 separate models (one per subject) - output dirs auto-derived from model name
  python training/finetune.py \
    --model Qwen/Qwen3-1.7B \
    --data_history data/train_history.json \
    --data_geography data/train_geography.json \
    --data_algebra data/train_algebra.json \
    --data_general data/train.json \
    --out adapters/
  # Creates: adapters/qwen3-1.7b-lora-algebra/
  #          adapters/qwen3-1.7b-lora-geography/
  #          adapters/qwen3-1.7b-lora-history/
  #          adapters/qwen3-1.7b-lora-general/

  # Train on a single subject - output dir auto-derived
  python training/finetune.py \
    --model Qwen/Qwen3-1.7B \
    --data_history data/train_history.json
  # Creates: adapters/qwen3-0.6b-lora/ (if using --data) or adapters/qwen3-0.6b-lora-history/ (if using --data_history)

  # Custom output directory (overrides auto-derivation)
  python training/finetune.py \
    --model Qwen/Qwen3-1.7B \
    --data data/train.json \
    --out adapters/my-custom-adapter

Dataset format expects JSON or JSONL files with keys: {"question": str, "answer": str}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _extract_model_name_for_adapter(model_name: str) -> str:
    """
    Extract a normalized model name for adapter directory naming.

    Examples:
        "Qwen/Qwen3-0.6B" -> "qwen3-0.6b"
        "Qwen/Qwen3-1.7B" -> "qwen3-1.7b"
        "Qwen3-0.6B" -> "qwen3-0.6b"
    """
    # Remove organization prefix if present (e.g., "Qwen/")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    # Convert to lowercase
    model_name = model_name.lower()

    return model_name


def _get_adapter_base_name(model_name: str) -> str:
    """
    Get the base name for adapter directories based on model name.

    Examples:
        "Qwen/Qwen3-0.6B" -> "qwen3-0.6b-lora"
        "Qwen/Qwen3-1.7B" -> "qwen3-1.7b-lora"
    """
    base_name = _extract_model_name_for_adapter(model_name)
    return f"{base_name}-lora"


class QAJsonlDataset(Dataset):
    def __init__(self, paths: List[str], tokenizer: AutoTokenizer, max_len: int = 1024, subject: str = None):
        """
        Initialize dataset from one or more JSON/JSONL files.

        Args:
            paths: List of paths to JSON or JSONL files (can be empty strings for missing files)
            tokenizer: Tokenizer to use
            max_len: Maximum sequence length
            subject: Optional subject name for subject-specific prompts
        """
        self.lines: List[Dict] = []
        self.tok = tokenizer
        self.max_len = max_len
        self.subject = subject

        # Load data from all provided paths
        for path in paths:
            if path and Path(path).exists():
                file_path = Path(path)
                file_count = 0

                # Handle JSON files (array format)
                if file_path.suffix.lower() == '.json':
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Handle both list and dict formats
                        if isinstance(data, list):
                            items = data
                        elif isinstance(data, dict) and 'data' in data:
                            items = data['data']
                        elif isinstance(data, dict):
                            # If it's a single object, wrap it in a list
                            items = [data]
                        else:
                            raise ValueError(
                                f"Unexpected JSON format in {file_path}")

                        # Process each item
                        for item in items:
                            if isinstance(item, dict):
                                # Validate required fields
                                if 'question' in item and 'answer' in item:
                                    if self.subject:
                                        item['_subject'] = self.subject
                                    item['_file_path'] = str(file_path)
                                    self.lines.append(item)
                                    file_count += 1
                                else:
                                    print(
                                        f"Warning: Skipping item without 'question' and 'answer' in {file_path}")

                        print(
                            f"Loaded {file_count} examples from {file_path} (JSON format)")
                    except json.JSONDecodeError as e:
                        print(f"Error: Invalid JSON in {file_path}: {e}")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

                # Handle JSONL files (one JSON object per line)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    # Validate required fields
                                    if 'question' in item and 'answer' in item:
                                        # Add subject info if available
                                        if self.subject:
                                            item['_subject'] = self.subject
                                        item['_file_path'] = str(file_path)
                                        self.lines.append(item)
                                        file_count += 1
                                    else:
                                        print(
                                            f"Warning: Skipping item without 'question' and 'answer' in {file_path}")
                                except json.JSONDecodeError:
                                    print(
                                        f"Warning: Skipping invalid JSON line in {file_path}")
                    print(
                        f"Loaded {file_count} examples from {file_path} (JSONL format)")
            elif path:
                print(f"Warning: Data file not found: {path}")

        if not self.lines:
            raise ValueError(f"No data loaded from paths: {paths}")

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.lines[idx]
        question = item['question']
        answer = item['answer']

        # Use subject-specific prompt format if subject is known
        if self.subject == "algebra":
            prompt = f"Question: {question}\nAnswer:"
        elif self.subject == "geography":
            prompt = f"Question: {question}\nAnswer:"
        elif self.subject == "history":
            prompt = f"Question: {question}\nAnswer:"
        else:
            # General/default format
            prompt = f"Question: {question}\nAnswer:"

        text = prompt + " " + answer
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )
        return enc


def train_single_model(
    model_name: str,
    data_file: str,
    output_dir: str,
    subject: str,
    epochs: int,
    lr: float,
    batch: int,
    device_map: str,
    quant: str,
    grad_ckpt: bool,
    offload_folder: str = None,
    int8_cpu_offload: bool = True,
) -> bool:
    """
    Train a single adapter model on a dataset.

    Returns:
        True if training succeeded, False otherwise
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="/app/models", use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine if CUDA is available and adjust accordingly
        is_cpu_only = not torch.cuda.is_available()

        # Handle quantization and device mapping - different paths for CPU vs GPU
        load_kwargs = {"trust_remote_code": True}

        if is_cpu_only:
            # CPU-only: no quantization, no device_map to avoid offloading warnings
            load_kwargs["dtype"] = torch.float32
            # Don't use device_map for CPU to avoid offloading logic
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir="/app/models",
                **load_kwargs
            )
            model = model.to("cpu")
        else:
            # GPU: setup quantization and device_map
            if quant == "4bit":
                qconf = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                load_kwargs["quantization_config"] = qconf
            elif quant == "8bit":
                qconf = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=int8_cpu_offload,
                )
                load_kwargs["quantization_config"] = qconf

            # Determine device_map - use explicit device for single GPU to avoid offloading
            if device_map == "auto":
                device_map_val = "cuda:0"  # Explicit device for single GPU
            else:
                device_map_val = device_map

            load_kwargs["device_map"] = device_map_val
            if offload_folder:
                load_kwargs["offload_folder"] = offload_folder

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir="/app/models",
                **load_kwargs
            )
            # Only prepare for kbit training if using quantization on GPU
            if quant in ["4bit", "8bit"]:
                model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        base_model = model
        if hasattr(model, 'base_model'):
            base_model = model.base_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        model_modules = [name for name, _ in base_model.named_modules()]
        available_targets = [m for m in target_modules if any(
            m in name for name in model_modules)]

        if not available_targets:
            print("Warning: Standard target modules not found. Trying to auto-detect...")
            for name in model_modules:
                if "proj" in name.lower() and ("q" in name.lower() or "k" in name.lower() or "v" in name.lower() or "o" in name.lower()):
                    module_name = name.split('.')[-1]
                    if module_name not in available_targets:
                        available_targets.append(module_name)

        if not available_targets:
            available_targets = target_modules
            print(
                f"Warning: Could not auto-detect target modules. Using default: {available_targets}")
        else:
            print(f"Using target modules: {available_targets}")

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=available_targets,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Apply LoRA to the model
        try:
            model = get_peft_model(model, lora_cfg)
        except Exception as e:
            print(
                f"Error applying LoRA with target_modules {available_targets}: {e}")
            print("Trying with default target modules...")
            lora_cfg.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            model = get_peft_model(model, lora_cfg)

        # Set model to training mode
        model.train()
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            model.base_model.model.train()
        if hasattr(model, 'base_model'):
            model.base_model.train()

        # Verify LoRA parameters are trainable
        trainable_after = sum(p.numel()
                              for p in model.parameters() if p.requires_grad)
        if trainable_after == 0:
            raise RuntimeError(
                "No trainable parameters found after LoRA setup!")

        # Enable gradient checkpointing
        if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        # Create dataset
        train_ds = QAJsonlDataset([data_file], tokenizer, subject=subject)
        print(f"Total training examples: {len(train_ds)}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=1,
            learning_rate=lr,
            num_train_epochs=epochs,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            bf16=not is_cpu_only and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=not is_cpu_only and torch.cuda.is_available(
            ) and not torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit" if not is_cpu_only else "adamw_torch",
            report_to=[],
            gradient_checkpointing=False,  # We handle it manually above
        )

        # Device placement
        if not is_cpu_only:
            uses_device_map = hasattr(
                model, 'hf_device_map') and model.hf_device_map is not None
            if not uses_device_map:
                device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(device)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )

        model.train()
        if not model.training:
            print("WARNING: Model is not in training mode! Forcing train mode...")
            model.train()

        trainer = Trainer(model=model, args=training_args,
                          train_dataset=train_ds, data_collator=data_collator)

        if hasattr(trainer.model, 'train'):
            trainer.model.train()

        torch.cuda.empty_cache()

        trainer.train()
        # MUST use the PEFT model's save
        model.save_pretrained(output_dir, safe_serialization=True)

        # Save tokenizer next to adapter
        tokenizer.save_pretrained(output_dir)

        print(
            f"✓ Successfully trained and saved {subject} adapter to {output_dir}")
        return True

    except Exception as e:
        print(f"✗ Failed to train {subject} adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fine-tune model with subject-specific datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train 4 separate models (one per subject) - saves to separate folders
  python training/finetune.py \\
    --data_history data/train_history.json \\
    --data_geography data/train_geography.json \\
    --data_algebra data/train_algebra.json \\
    --data_general data/train.json \\
    --out adapters/
  # This creates: adapters/qwen3-1.7b-lora-history/
  #              adapters/qwen3-1.7b-lora-geography/
  #              adapters/qwen3-1.7b-lora-algebra/
  #              adapters/qwen3-1.7b-lora-general/

  # Train on single subject only
  python training/finetune.py --data_history data/train_history.json --out adapters/history

  # Use single data file (backward compatible)
  python training/finetune.py --data data/train.json --out adapters/general
        """
    )
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="Base model to fine-tune")

    # Subject-specific data arguments
    ap.add_argument("--data",
                    help="Single data file (backward compatible)", default="./data/train.json")
    ap.add_argument("--data_history",
                    help="History training data (JSON)", default="./data/train_history.json")
    ap.add_argument("--data_geography",
                    help="Geography training data (JSON)", default="./data/train_geography.json")
    ap.add_argument("--data_algebra",
                    help="Algebra training data (JSON)", default="./data/train_algebra.json")
    # ap.add_argument("--data_general",
    # help="General training data (JSON)", default="./data/train.json")
    ap.add_argument("--data_chinese",
                    help="Chinese training data (JSON)", default="./data/train_chinese.json")
    ap.add_argument("--out", default=None,
                    help="Output directory for adapter (auto-derived from model name if not specified)")
    ap.add_argument("--epochs", type=int, default=4,
                    help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    ap.add_argument("--batch", type=int, default=1,
                    help="Batch size per device")
    ap.add_argument("--device_map", default="auto",
                    help="Device mapping strategy")
    ap.add_argument("--offload_folder", default=None,
                    help="Folder for CPU offloading")
    ap.add_argument(
        "--quant", choices=["4bit", "8bit"], default="4bit", help="Quantization type")
    ap.add_argument("--int8_cpu_offload", action="store_true",
                    default=True, help="Enable INT8 CPU offload")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="Enable gradient checkpointing", default=True)
    ap.add_argument("--subject", choices=["history", "geography", "algebra", "general", "chinese", None],
                    default=None, help="Subject type for single-subject training (auto-detected if multiple files)")
    args = ap.parse_args()

    # Determine which data files to use
    subject_configs = []

    # Check for subject-specific files
    if args.data_algebra and Path(args.data_algebra).exists():
        subject_configs.append(("algebra", args.data_algebra))
    if args.data_geography and Path(args.data_geography).exists():
        subject_configs.append(("geography", args.data_geography))
    if args.data_history and Path(args.data_history).exists():
        subject_configs.append(("history", args.data_history))
    if args.data_chinese and Path(args.data_chinese).exists():
        subject_configs.append(("chinese", args.data_chinese))
    if args.data_general and Path(args.data_general).exists():
       subject_configs.append(("general", args.data_general))
    # Check if we should train multiple models (4 separate models)
    train_multiple = len(subject_configs) > 1

    # If no subject-specific files, use --data for single model training
    if not subject_configs:
        if args.data and Path(args.data).exists():
            # Determine subject from filename or use "general"
            data_path = Path(args.data)
            if "algebra" in data_path.stem.lower():
                subject = "algebra"
            elif "geography" in data_path.stem.lower():
                subject = "geography"
            elif "history" in data_path.stem.lower():
                subject = "history"
            elif "chinese" in data_path.stem.lower():
                subject = "chinese"
            else:
                subject = args.subject or "general"
            subject_configs = [(subject, args.data)]
        else:
            raise ValueError(
                "No data files found. Use --data or subject-specific flags "
                "(--data_history, --data_geography, --data_algebra, --data_general, --data_chinese)"
            )

    # Get adapter base name from model (used for multi-model training)
    adapter_base_name = _get_adapter_base_name(args.model)

    # Auto-derive output directory from model name if not specified
    if args.out is None:
        # For single model, use base name; for multiple, will use base_out as parent
        args.out = f"./adapters/{adapter_base_name}"

    # Determine base output directory
    base_out = Path(args.out)

    if train_multiple:
        # Train 4 separate models
        print(f"\n{'='*60}")
        print("TRAINING 4 SEPARATE MODELS (one per subject)")
        print(f"{'='*60}")
        print(f"Base model: {args.model}")
        print(f"Base output directory: {base_out}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size: {args.batch}")
        print(f"Subjects to train: {[s[0] for s in subject_configs]}")
        print(f"{'='*60}\n")

        # Get adapter base name from model
        adapter_base_name = _get_adapter_base_name(args.model)

        results = {}
        for subject, data_file in subject_configs:
            output_dir = base_out / f"{adapter_base_name}-{subject}"
            print(f"\n{'='*60}")
            print(f"Training {subject.upper()} model")
            print(f"{'='*60}")
            print(f"Data file: {data_file}")
            print(f"Output: {output_dir}")
            print()

            success = train_single_model(
                model_name=args.model,
                data_file=data_file,
                output_dir=str(output_dir),
                subject=subject,
                epochs=args.epochs,
                lr=args.lr,
                batch=args.batch,
                device_map=args.device_map,
                quant=args.quant,
                grad_ckpt=args.grad_ckpt,
                offload_folder=args.offload_folder,
                int8_cpu_offload=args.int8_cpu_offload,
            )
            results[subject] = success

        # Print summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        for subject, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{subject.upper():12} : {status}")
        print(f"{'='*60}\n")

        if not all(results.values()):
            import sys
            sys.exit(1)

    else:
        # Train single model (backward compatible)
        subject, data_file = subject_configs[0]
        print(f"\n=== Training Configuration ===")
        print(f"Subject: {subject}")
        print(f"Data file: {data_file}")
        print(f"Output: {args.out}/single_model")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Batch size: {args.batch}")
        print("=" * 30 + "\n")

        success = train_single_model(
            model_name=args.model,
            data_file=data_file,
            output_dir=args.out+"/single_model",
            subject=subject,
            epochs=args.epochs,
            lr=args.lr,
            batch=args.batch,
            device_map=args.device_map,
            quant=args.quant,
            grad_ckpt=args.grad_ckpt,
            offload_folder=args.offload_folder,
            int8_cpu_offload=args.int8_cpu_offload,
        )

        if not success:
            import sys
            sys.exit(1)


if __name__ == "__main__":
    main()

