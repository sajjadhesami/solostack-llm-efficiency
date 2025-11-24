from __future__ import annotations
import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig

import os
import gc
import json
from typing import Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force strict offline mode for Hugging Face - SET THESE AT THE VERY TOP
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Block all internet access at the socket level
os.environ["TRANSFORMERS_NO_INTERNET"] = "1"
os.environ["PEFT_NO_INTERNET"] = "1"

# Handle imports - try relative first, fall back to absolute
try:
    from .config import (
        CACHE_DIR,
        DEFAULT_MODEL,
        PEFT_ADAPTER_ALGEBRA,
        PEFT_ADAPTER_GEOGRAPHY,
        PEFT_ADAPTER_HISTORY,
        PEFT_ADAPTER_CHINESE,
        PEFT_ADAPTER_GENERAL,
        USE_4BIT,
        DEVICE_MAP,
        BATCH_SIZE,
        MAX_NEW_TOKENS,
        TEMPERATURE,
        TOP_P,
        MAX_INPUT_TOKENS,
        MAX_TOKENS_MULTIPLIER,
        MIN_NEW_TOKENS,
        REPETITION_PENALTY,
        TOP_P_CAP,
        TOP_K,
        RAG_KNOWLEDGE_BASE_DIR,
        RAG_TOP_K,
        RAG_ENABLED_SUBJECTS,
    )
    from .prompt import build_chat_messages_with_subject, build_plain_prompt
    from .utils import chunk_iter
    from .rag import retrieve_context
except ImportError:
    # Fallback for absolute imports if relative fails
    from inferencePipeline.config import (
        CACHE_DIR,
        DEFAULT_MODEL,
        PEFT_ADAPTER_ALGEBRA,
        PEFT_ADAPTER_GEOGRAPHY,
        PEFT_ADAPTER_HISTORY,
        PEFT_ADAPTER_CHINESE,
        PEFT_ADAPTER_GENERAL,
        USE_4BIT,
        DEVICE_MAP,
        BATCH_SIZE,
        MAX_NEW_TOKENS,
        TEMPERATURE,
        TOP_P,
        MAX_INPUT_TOKENS,
        MAX_TOKENS_MULTIPLIER,
        MIN_NEW_TOKENS,
        REPETITION_PENALTY,
        TOP_P_CAP,
        TOP_K,
        RAG_KNOWLEDGE_BASE_DIR,
        RAG_TOP_K,
        RAG_ENABLED_SUBJECTS,
    )
    from inferencePipeline.prompt import build_chat_messages_with_subject, build_plain_prompt
    from inferencePipeline.utils import chunk_iter
    from inferencePipeline.rag import retrieve_context


# Precompile regex patterns for better performance
_THINK_PATTERN = re.compile(
    r"<\s*think[^>]*>[\s\S]*?<\s*/\s*think\s*>", re.IGNORECASE)
_THINK_ORPHAN_PATTERN = re.compile(r"\b\w+\s*<\s*/\s*think\s*>", re.IGNORECASE)
_THINK_TAG_PATTERN = re.compile(r"<\s*/?\s*think[^>]*>", re.IGNORECASE)
_ROLE_MARKER_PATTERN = re.compile(
    r"(?:^|\n)\s*assistant\b[:\s-]*", re.IGNORECASE)
_SYSTEM_USER_PATTERN = re.compile(
    r"^(system|user)\s*:?.*?(\n|$)", re.IGNORECASE)
_ASSISTANT_PATTERN = re.compile(r"^(assistant)\s*:?", re.IGNORECASE)
_CONTROL_CHARS_PATTERN = re.compile(
    r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_PUNCTUATION_PATTERN = re.compile(r"^[\-:–—•\s]+")


def _clean_answer(text: str) -> str:
    """
    Safe, Qwen-optimized answer cleaner.
    - Removes chain-of-thought styled tags (<think>, <analysis>, etc.)
    - Removes leaked reasoning or system echoes
    - Keeps only the final, user-facing answer
    - Prevents duplicate or repeated lines
    - Never removes short answers by accident
    """

    if not text:
        return ""

    original = text

    # Normalize newlines
    text = text.replace("\r", "")

    # Remove ALL internal tags + their content
    # Qwen often uses <think> ... </think> or <analysis> ... </analysis>
    inner_re = re.compile(
        r"<(think|analysis|internal|reasoning|scratchpad)[^>]*>.*?</\1>",
        re.IGNORECASE | re.DOTALL,
    )
    text = inner_re.sub("", text)

    # Remove self-closing bogus tags
    text = re.sub(
        r"</?(think|analysis|internal|reasoning|scratchpad)[^>]*>", "", text, flags=re.IGNORECASE)

    # Remove repeated "Answer:" prefixes
    text = re.sub(r"(?i)^\s*answer\s*:\s*", "", text).strip()

    # If model echoes system prompt, cut everything before the last "Answer:" or question mark
    # Qwen sometimes outputs system prompt accidentally
    if "Answer:" in text:
        candidates = text.split("Answer:")
        text = candidates[-1].strip()

    # Remove system or prompt fragments Qwen tends to echo
    bad_prefixes = [
        "Instruction:", "System:", "You are", "Goal:", "Guidelines:",
        "Relevant Context:", "<system>", "</system>"
    ]
    lines = text.split("\n")
    filtered = []
    for line in lines:
        if not any(line.strip().startswith(bp) for bp in bad_prefixes):
            filtered.append(line)
    text = "\n".join(filtered).strip()

    # Remove hallucinated XML/JSON block residues
    text = re.sub(r"<[^>]+>", "", text)

    # Remove repeating duplicate sentences (common on Qwen)
    seen = set()
    deduped = []
    for line in text.split("\n"):
        L = line.strip()
        if not L:
            continue
        if L not in seen:
            deduped.append(L)
            seen.add(L)
    text = "\n".join(deduped).strip()

    # Trim whitespace and stray punctuation
    text = text.strip(" \t\r\n.,;")

    # Fallback: if cleaning removed everything, return original minimal content
    if not text:
        return original.strip()

    return text


def _detect_dtype() -> torch.dtype:
    """Detect the appropriate torch dtype for the current hardware."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _maybe_int4_kwargs(device_map: str) -> dict:
    """Configure 4-bit quantization if available and enabled."""
    # 4-bit quantization requires CUDA - skip on CPU-only systems
    is_cpu = device_map == "cpu" or not torch.cuda.is_available()
    if not USE_4BIT or is_cpu:
        return {"dtype": _detect_dtype()}

    try:
        # Check if bitsandbytes is available
        import bitsandbytes as bnb
        print("✓ Using 4-bit quantization with bitsandbytes")

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return {"quantization_config": qconf}
    except ImportError:
        print("⚠️ bitsandbytes not available, falling back to standard dtype")
        return {"dtype": _detect_dtype()}


def _verify_model_cache(model_name: str) -> bool:
    """Verify that the model is available in the local cache."""
    from transformers.utils.hub import cached_file

    try:
        # Check if model files exist in cache
        config_path = cached_file(
            model_name,
            "config.json",
            cache_dir=CACHE_DIR,
            local_files_only=True
        )
        if config_path and os.path.exists(config_path):
            print(
                f"✓ Found model config in cache: {os.path.basename(config_path)}")
            return True
        else:
            print(f"✗ Model config not found in cache for: {model_name}")
            return False
    except Exception as e:
        print(f"✗ Error checking model cache for {model_name}: {e}")
        return False


def _load_peft_adapter_directly(base_model, adapter_path: str):
    """
    Fully offline adapter loader that uses PEFT's config loader
    (which does NOT require internet when files exist locally).
    """

    print(f"Loading PEFT adapter (offline) from: {adapter_path}")

    # Load PEFT config (this constructs the correct config class)
    try:
        adapter_config = PeftConfig.from_pretrained(
            adapter_path,
            local_files_only=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load adapter config: {e}")

    # Create PEFT model wrapper
    try:
        peft_model = PeftModel(base_model, adapter_config)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PeftModel: {e}")

    # Load actual LoRA weights
    try:
        peft_model.load_adapter(
            adapter_path,
            adapter_name="default",
            is_trainable=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load adapter weights: {e}")

    print("✓ Offline PEFT adapter loaded successfully")
    return peft_model


def _load_base_model(model_name: str, compile_model: bool = False):
    """Load the base model with quantization and optimizations for offline use."""
    print(f"Loading base model: {model_name}")

    # Verify model is in cache before attempting to load
    if not _verify_model_cache(model_name):
        raise FileNotFoundError(
            f"Model {model_name} not found in local cache. "
            f"Please ensure the model is downloaded to: {CACHE_DIR}"
        )

    # Enable cuDNN benchmark for better GPU performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        # Optimize CPU threading for better performance
        torch.set_num_threads(min(12, os.cpu_count() or 4))

    # Determine device for CPU-only systems
    is_cpu_only = not torch.cuda.is_available() or DEVICE_MAP == "cpu"
    model_device_map = "cpu" if is_cpu_only else DEVICE_MAP

    # Get quantization kwargs (will disable quantization for CPU)
    quant_kwargs = _maybe_int4_kwargs(model_device_map)

    # Try to use Flash Attention 2 for faster inference (if available and on GPU)
    attn_implementation = None
    if not is_cpu_only:
        try:
            # Check if flash-attn is available
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
            print("✓ Using Flash Attention 2 for faster inference")
        except ImportError:
            # Fall back to default (usually "sdpa" which is also fast)
            attn_implementation = "sdpa"
            print("✓ Using SDPA attention for faster inference")

    model_kwargs = {
        "cache_dir": CACHE_DIR,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": True,  # Critical for offline operation
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    # For CPU-only, use explicit device placement
    if is_cpu_only:
        print("✓ Loading model on CPU")
        model_kwargs["dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs)
        model = model.to("cpu")
    else:
        print(f"✓ Loading model on {model_device_map}")
        model_kwargs["device_map"] = model_device_map
        model_kwargs.update(quant_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

    # Set the model to evaluation mode
    model.eval()

    # Compile model only if requested
    if compile_model:
        try:
            if hasattr(torch, 'compile'):
                if torch.cuda.is_available():
                    print("✓ Compiling model for GPU (reduce-overhead mode)")
                    model = torch.compile(
                        model, mode='reduce-overhead', fullgraph=False)
                else:
                    print("✓ Compiling model for CPU (default mode)")
                    model = torch.compile(
                        model, mode='default', fullgraph=False)
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")

    # Disable gradient computation globally for extra safety and speed
    for param in model.parameters():
        param.requires_grad = False

    print(f"✓ Successfully loaded base model: {model_name}")
    return model


def _load_tokenizer(model_name: str):
    """Load the tokenizer for the model in offline mode."""
    print(f"Loading tokenizer for: {model_name}")

    # Verify tokenizer is in cache
    if not _verify_model_cache(model_name):
        raise FileNotFoundError(
            f"Tokenizer for {model_name} not found in local cache."
        )

    try:
        # Try fast tokenizer first
        tok = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True  # Critical for offline
        )
        print("✓ Using fast tokenizer")
    except Exception as e:
        print(f"⚠️ Fast tokenizer failed, falling back to slow tokenizer: {e}")
        try:
            tok = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR,
                use_fast=False,
                trust_remote_code=True,
                local_files_only=True  # Critical for offline
            )
            print("✓ Using slow tokenizer")
        except Exception as e2:
            raise RuntimeError(f"Failed to load tokenizer: {e2}")

    # Set padding side to left for more efficient batch generation
    if hasattr(tok, 'padding_side'):
        tok.padding_side = 'left'

    # Ensure pad token is set
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        print("✓ Set pad token to eos token")

    print(f"✓ Successfully loaded tokenizer")
    return tok


def _validate_adapter_directory(adapter_path: str) -> bool:
    """Validate that an adapter directory contains the required files."""
    if not adapter_path or not os.path.isdir(adapter_path):
        print(f"✗ Adapter directory does not exist: {adapter_path}")
        return False

    required_files = ["adapter_config.json"]
    optional_files = ["adapter_model.bin", "adapter_model.safetensors"]

    print(f"Checking adapter directory: {adapter_path}")

    # Check for required files
    missing_required = []
    for file in required_files:
        file_path = os.path.join(adapter_path, file)
        if os.path.exists(file_path):
            print(f"  ✓ Found {file}")
        else:
            print(f"  ✗ Missing required file: {file}")
            missing_required.append(file)

    if missing_required:
        print(f"✗ Adapter is missing required files: {missing_required}")
        return False

    # Check for at least one model file
    found_model = False
    for file in optional_files:
        file_path = os.path.join(adapter_path, file)
        if os.path.exists(file_path):
            print(f"  ✓ Found {file}")
            found_model = True
            break

    if not found_model:
        print(
            f"✗ Adapter is missing model weights (need one of: {', '.join(optional_files)})")
        return False

    print(f"✓ Adapter directory is valid")
    return True


def _load_base_model_for_adapter(model_name: str):
    """Load base model specifically for applying PEFT adapters in offline mode."""
    print(f"Loading base model for adapter: {model_name}")

    # Verify model is in cache
    if not _verify_model_cache(model_name):
        raise FileNotFoundError(
            f"Base model {model_name} not found in cache for adapter loading"
        )

    # Enable cuDNN benchmark for better GPU performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.set_num_threads(min(12, os.cpu_count() or 4))

    # Determine device
    is_cpu_only = not torch.cuda.is_available() or DEVICE_MAP == "cpu"
    target_device = "cpu" if is_cpu_only else (
        "cuda:0" if torch.cuda.is_available() else "cpu")

    # Get quantization kwargs
    quant_kwargs = _maybe_int4_kwargs(target_device)

    # Try to use Flash Attention 2
    attn_implementation = None
    if not is_cpu_only:
        try:
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
        except ImportError:
            attn_implementation = "sdpa"

    # Build model kwargs
    model_kwargs = {
        "cache_dir": CACHE_DIR,
        "trust_remote_code": True,
        "local_files_only": True,  # Critical for offline
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    # Load model with quantization if available
    if is_cpu_only:
        print("✓ Loading base model on CPU for adapter")
        model_kwargs["dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs)
        model = model.to("cpu")
    else:
        print(f"✓ Loading base model on {target_device} for adapter")
        model_kwargs.update(quant_kwargs)
        model_kwargs["device_map"] = target_device
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs)

    # Set to evaluation mode
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    print(f"✓ Successfully loaded base model for adapter")
    return model


def _load_model_with_adapter(model_name: str, peft_adapter_path: str):
    """Load a model with a PEFT adapter applied in offline mode."""
    # Determine if we're on CPU
    is_cpu_only = not torch.cuda.is_available() or DEVICE_MAP == "cpu"

    if peft_adapter_path and os.path.isdir(peft_adapter_path):
        # Validate adapter directory before attempting to load
        if not _validate_adapter_directory(peft_adapter_path):
            raise FileNotFoundError(
                f"Adapter directory is missing required files: {peft_adapter_path}")

        try:
            # Load base model from scratch
            print(f"Loading base model: {model_name}")
            base_model = _load_base_model_for_adapter(model_name)
            print(f"✓ Base model loaded")

            # Apply PEFT adapter using our direct loading method
            print(f"Loading PEFT adapter from: {peft_adapter_path}")
            # Ensure absolute path
            if not os.path.isabs(peft_adapter_path):
                peft_adapter_path = os.path.abspath(peft_adapter_path)

            # Use our direct loading method that bypasses online checks
            model_with_adapter = _load_peft_adapter_directly(
                base_model, peft_adapter_path)

            # Compile the model with adapter for faster inference
            try:
                if hasattr(torch, 'compile'):
                    if torch.cuda.is_available():
                        model_with_adapter = torch.compile(
                            model_with_adapter, mode='reduce-overhead', fullgraph=False)
                    else:
                        model_with_adapter = torch.compile(
                            model_with_adapter, mode='default', fullgraph=False)
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")

            if is_cpu_only:
                model_with_adapter = model_with_adapter.to("cpu")
            return model_with_adapter
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model with adapter from {peft_adapter_path}: {e}")

    # If no adapter path, just load the base model
    print("No adapter path provided, loading base model only")
    return _load_base_model(model_name, compile_model=True)


def _check_all_models_available(model_name: str, subject_adapters: Dict[str, str]) -> bool:
    """Check if all required models and adapters are available offline."""
    print("Checking model availability for offline operation...")

    # Check base model
    if not _verify_model_cache(model_name):
        print(f"✗ Base model {model_name} not found in cache")
        return False
    print(f"✓ Base model {model_name} available")

    # Check tokenizer
    try:
        _load_tokenizer(model_name)
        print("✓ Tokenizer available")
    except Exception as e:
        print(f"✗ Tokenizer not available: {e}")
        return False

    # Check adapters
    available_adapters = 0
    for subject, adapter_path in subject_adapters.items():
        if _validate_adapter_directory(adapter_path):
            print(f"✓ {subject} adapter available")
            available_adapters += 1
        else:
            print(f"✗ {subject} adapter not available")

    if available_adapters == 0:
        print("⚠️ No adapters available, will use base model only")

    print(f"✓ All required models available for offline operation")
    return True


def build_pipeline(model_name: str | None = None) -> Callable[[List[Dict]], List[Dict]]:
    """Build the inference pipeline for fully offline operation."""
    model_name = model_name or DEFAULT_MODEL

    print("=" * 60)
    print("INITIALIZING OFFLINE INFERENCE PIPELINE")
    print("=" * 60)

    # Map subject names to adapter paths
    subject_adapters = {
        "algebra": PEFT_ADAPTER_ALGEBRA,
        "geography": PEFT_ADAPTER_GEOGRAPHY,
        "history": PEFT_ADAPTER_HISTORY,
        "chinese": PEFT_ADAPTER_CHINESE,
        "general": PEFT_ADAPTER_GENERAL,
    }

    # Verify all models are available offline
    if not _check_all_models_available(model_name, subject_adapters):
        raise RuntimeError(
            "Required models not found in local cache. "
            "Please ensure all models and adapters are downloaded before running in offline mode."
        )

    # Load base tokenizer
    print("Loading base tokenizer...")
    base_tokenizer = _load_tokenizer(model_name)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    if hasattr(base_tokenizer, 'padding_side'):
        base_tokenizer.padding_side = 'left'

    is_chat_model = hasattr(base_tokenizer, "apply_chat_template")

    # Check which adapters exist and validate them
    available_subjects = {}
    print("Validating adapter directories...")
    for subject, adapter_path in subject_adapters.items():
        if _validate_adapter_directory(adapter_path):
            available_subjects[subject] = True
            print(f"  ✓ {subject} adapter is available")
        else:
            available_subjects[subject] = False
            print(f"  ✗ {subject} adapter not available")

    # Check if at least one adapter exists
    if not any(available_subjects.values()):
        print("⚠️ No subject-specific adapters found, will use base model only")

    # Configure generation parameters
    gen_cfg = GenerationConfig()

    # Adjust generation parameters
    effective_max_tokens = int(MAX_NEW_TOKENS * MAX_TOKENS_MULTIPLIER)
    gen_cfg.max_new_tokens = effective_max_tokens
    gen_cfg.min_new_tokens = MIN_NEW_TOKENS

    # Use sampling mode
    gen_cfg.do_sample = TEMPERATURE > 0
    gen_cfg.temperature = TEMPERATURE
    gen_cfg.top_p = min(TOP_P, TOP_P_CAP)
    gen_cfg.top_k = TOP_K
    gen_cfg.repetition_penalty = REPETITION_PENALTY

    gen_cfg.eos_token_id = base_tokenizer.eos_token_id
    gen_cfg.bos_token_id = getattr(base_tokenizer, "bos_token_id", None)
    gen_cfg.pad_token_id = base_tokenizer.eos_token_id
    gen_cfg.use_cache = True
    gen_cfg.output_scores = False
    gen_cfg.output_attentions = False
    gen_cfg.output_hidden_states = False
    gen_cfg.return_dict_in_generate = False

    print("✓ Generation configuration ready")
    print("✓ Pipeline initialized successfully for offline operation")
    print("=" * 60)

    def answer_fn(questions: List[Dict]) -> List[Dict]:
        """Process questions in fully offline mode."""
        if not questions:
            return []

        # Pre-allocate output list
        outputs: List[Dict] = [None] * len(questions)
        question_to_index = {id(q): i for i, q in enumerate(questions)}

        # Step 1: Extract subjects from question JSON
        print("Extracting subjects from questions...")
        all_subjects = []
        for q in questions:
            # Get subject from question JSON
            subject = q.get("subject", "").lower().strip()

            # Normalize subject
            if "algebra" in subject:
                subject = "algebra"
            elif "geography" in subject:
                subject = "geography"
            elif "history" in subject:
                subject = "history"
            elif "chinese" in subject:
                subject = "chinese"
            else:
                subject = "general"  # Default to general

            # Map to available adapter
            if subject not in available_subjects or not available_subjects.get(subject):
                if available_subjects.get("general", False):
                    subject = "general"
                else:
                    # Use first available subject adapter
                    available = [
                        s for s, avail in available_subjects.items() if avail]
                    if available:
                        subject = available[0]
                    # If no adapters available, keep original subject

            all_subjects.append(subject)

        # Step 2: Group questions by subject
        questions_by_subject = {}
        for q, subject in zip(questions, all_subjects):
            if subject not in questions_by_subject:
                questions_by_subject[subject] = []
            questions_by_subject[subject].append(q)

        print(
            f"Questions grouped by subject: {[(s, len(qs)) for s, qs in questions_by_subject.items()]}")

        # Step 3: Process each subject group
        general_adapter_path = subject_adapters.get("general")
        has_general = general_adapter_path and os.path.isdir(
            general_adapter_path)

        for subject, subject_questions in questions_by_subject.items():
            # Check if adapter exists for this subject
            adapter_path = subject_adapters.get(subject)
            use_general_fallback = False

            if not adapter_path or not os.path.isdir(adapter_path):
                # Subject-specific adapter not found, use general as fallback
                if has_general:
                    print(
                        f"⚠️ No adapter for subject '{subject}', using general adapter for {len(subject_questions)} questions")
                    adapter_path = general_adapter_path
                    use_general_fallback = True
                else:
                    print(
                        f"✗ No adapter for subject '{subject}' and general adapter not available, skipping {len(subject_questions)} questions")
                    # Mark these questions as failed
                    for q in subject_questions:
                        idx = question_to_index[id(q)]
                        outputs[idx] = {"questionID": q["questionID"],
                                        "answer": "Error: Model not available"}
                    continue

            # Load model for this subject on-demand
            model_label = "general (fallback)" if use_general_fallback else subject
            print(
                f"Loading {model_label} model for {len(subject_questions)} {subject} questions...")

            try:
                # Clear GPU cache before loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Load model on-demand
                model = _load_model_with_adapter(model_name, adapter_path)

                # Move to GPU if available for faster inference
                if torch.cuda.is_available() and DEVICE_MAP != "cpu":
                    model = model.to("cuda:0")
                    torch.cuda.empty_cache()

                print(f"✓ Loaded {model_label} model")
            except Exception as e:
                print(f"✗ Failed to load {model_label} model: {e}")
                # Mark these questions as failed
                for q in subject_questions:
                    idx = question_to_index[id(q)]
                    outputs[idx] = {"questionID": q["questionID"],
                                    "answer": f"Error: Failed to load model - {str(e)}"}
                continue

            tokenizer = base_tokenizer

            print(
                f"Processing {len(subject_questions)} {subject} questions...")

            # Check if RAG is enabled for this subject
            use_rag = subject.lower() in RAG_ENABLED_SUBJECTS
            if use_rag:
                print(
                    f"✓ RAG enabled for {subject} - retrieving context from knowledge base")

            # Process in batches
            for batch in chunk_iter(subject_questions, BATCH_SIZE):
                # Retrieve RAG context for each question if enabled
                contexts = []
                if use_rag:
                    for q in batch:
                        try:
                            context = retrieve_context(
                                q["question"],
                                subject,
                                RAG_KNOWLEDGE_BASE_DIR,
                                RAG_TOP_K
                            )
                            contexts.append(context)
                        except Exception as e:
                            print(
                                f"⚠️ RAG retrieval failed for question {q['questionID']}: {e}")
                            contexts.append("")
                else:
                    contexts = [""] * len(batch)

                # Build prompts with RAG context
                if is_chat_model:
                    prompts = [
                        tokenizer.apply_chat_template(
                            build_chat_messages_with_subject(
                                q["question"], subject, context=ctx),
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for q, ctx in zip(batch, contexts)
                    ]
                else:
                    prompts = [
                        build_plain_prompt(q["question"], subject, context=ctx)
                        for q, ctx in zip(batch, contexts)
                    ]

                # Tokenize
                try:
                    enc = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_INPUT_TOKENS,
                        return_attention_mask=True,
                    )

                    # Move to device
                    try:
                        model_device = next(model.parameters()).device
                    except (StopIteration, AttributeError):
                        model_device = "cuda" if torch.cuda.is_available() else "cpu"

                    enc = {k: v.to(model_device, non_blocking=True)
                           for k, v in enc.items()}

                    # Generate
                    with torch.inference_mode():
                        gen = model.generate(**enc, generation_config=gen_cfg)

                    # Decode
                    prompt_lens = enc["attention_mask"].sum(dim=1)
                    generated_sequences = []
                    for i in range(len(batch)):
                        prompt_len = int(prompt_lens[i].item())
                        answer_ids = gen[i][prompt_len:].cpu()
                        generated_sequences.append(answer_ids)

                    if generated_sequences:
                        decoded_texts = tokenizer.batch_decode(
                            generated_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        decoded_texts = [text.strip()
                                         for text in decoded_texts]
                    else:
                        decoded_texts = [""] * len(batch)

                    # Store results
                    for i, q in enumerate(batch):
                        answer = _clean_answer(decoded_texts[i])
                        idx = question_to_index[id(q)]
                        outputs[idx] = {
                            "questionID": q["questionID"], "answer": answer}

                except Exception as e:
                    print(f"✗ Batch processing failed: {e}")
                    # Mark batch questions as failed
                    for i, q in enumerate(batch):
                        idx = question_to_index[id(q)]
                        outputs[idx] = {"questionID": q["questionID"],
                                        "answer": f"Error: Processing failed - {str(e)}"}

            # Free model memory after processing all questions for this subject
            print(f"✓ Completed {subject} questions, freeing model memory...")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()

        # Ensure all questions have responses
        for i, output in enumerate(outputs):
            if output is None:
                outputs[i] = {"questionID": questions[i]["questionID"],
                              "answer": "Error: No response generated"}

        print(f"✓ Completed processing all {len(questions)} questions")
        return outputs

    return answer_fn


# Utility function to verify offline setup
def verify_offline_setup(model_name: str = None) -> Dict[str, bool]:
    """Verify that all required components are available for offline operation."""
    model_name = model_name or DEFAULT_MODEL

    results = {
        "base_model": False,
        "tokenizer": False,
        "adapters": {}
    }

    print("Verifying offline setup...")

    # Check base model
    try:
        results["base_model"] = _verify_model_cache(model_name)
    except Exception as e:
        print(f"Base model check failed: {e}")

    # Check tokenizer
    try:
        _load_tokenizer(model_name)
        results["tokenizer"] = True
    except Exception as e:
        print(f"Tokenizer check failed: {e}")

    # Check adapters
    subject_adapters = {
        "algebra": PEFT_ADAPTER_ALGEBRA,
        "geography": PEFT_ADAPTER_GEOGRAPHY,
        "history": PEFT_ADAPTER_HISTORY,
        "chinese": PEFT_ADAPTER_CHINESE,
        "general": PEFT_ADAPTER_GENERAL,
    }

    for subject, adapter_path in subject_adapters.items():
        results["adapters"][subject] = _validate_adapter_directory(
            adapter_path)

    # Print summary
    print("\n" + "=" * 50)
    print("OFFLINE SETUP VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Base Model: {'✓' if results['base_model'] else '✗'}")
    print(f"Tokenizer: {'✓' if results['tokenizer'] else '✗'}")
    print("Adapters:")
    for subject, available in results["adapters"].items():
        print(f"  {subject}: {'✓' if available else '✗'}")
    print("=" * 50)

    return results
