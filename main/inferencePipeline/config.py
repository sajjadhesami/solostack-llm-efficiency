import os
import pathlib

# Force offline mode using Hugging Face's built-in offline support
# Based on: https://stackoverflow.com/questions/75110981/sslerror-httpsconnectionpoolhost-huggingface-co-port-443-max-retries-exce
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

my_path = pathlib.Path(__file__).parent.resolve()

# Where models are cached by the evaluation environment
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/app/models")

# Default model can be overridden via env var MODEL_NAME
DEFAULT_MODEL = os.environ.get(
    "MODEL_NAME",
    # Use a small, instruction-tuned model from the allowed list by default
    "Qwen/Qwen3-1.7B",
)


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


# Extract adapter base name from model
_ADAPTER_BASE_NAME = _get_adapter_base_name(DEFAULT_MODEL)

# Optional PEFT adapter path for fine-tuned LoRA weights (if present locally)
# Default single adapter path (backward compatibility)
PEFT_ADAPTER_PATH = os.environ.get(
    "PEFT_ADAPTER_PATH", os.path.join(str(my_path), "adapters", _ADAPTER_BASE_NAME))

# Subject-specific adapter paths
# Base directory for subject adapters
ADAPTERS_BASE_DIR = os.environ.get(
    "ADAPTERS_BASE_DIR", os.path.join(str(my_path), "adapters"))

# Individual subject adapter paths (auto-derived from model name)
PEFT_ADAPTER_ALGEBRA = os.environ.get(
    "PEFT_ADAPTER_ALGEBRA", os.path.join(ADAPTERS_BASE_DIR, f"{_ADAPTER_BASE_NAME}", f"{_ADAPTER_BASE_NAME}-algebra"))
PEFT_ADAPTER_GEOGRAPHY = os.environ.get(
    "PEFT_ADAPTER_GEOGRAPHY", os.path.join(ADAPTERS_BASE_DIR, f"{_ADAPTER_BASE_NAME}", f"{_ADAPTER_BASE_NAME}-geography"))
PEFT_ADAPTER_HISTORY = os.environ.get(
    "PEFT_ADAPTER_HISTORY", os.path.join(ADAPTERS_BASE_DIR, f"{_ADAPTER_BASE_NAME}", f"{_ADAPTER_BASE_NAME}-history"))
PEFT_ADAPTER_CHINESE = os.environ.get(
    "PEFT_ADAPTER_CHINESE", os.path.join(ADAPTERS_BASE_DIR, f"{_ADAPTER_BASE_NAME}", f"{_ADAPTER_BASE_NAME}-chinese"))
PEFT_ADAPTER_GENERAL = os.environ.get(
    "PEFT_ADAPTER_GENERAL", os.path.join(ADAPTERS_BASE_DIR, f"{_ADAPTER_BASE_NAME}", f"{_ADAPTER_BASE_NAME}-general"))

# Efficiency settings
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"
# Batch size optimized for Tesla T4 (16GB) with 4-bit quantized Qwen3-1.7B
# With 4-bit quantization: model ~1-2GB, leaving ~14-15GB for batches
# Recommended: 32-48 (safe), 48-64 (optimal), 64+ (aggressive, test first)
# Adjust based on actual memory usage and input sequence lengths
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "48"))
# Increased to prevent truncation and allow complete answers
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
# Temperature for sampling (higher = more exploration)
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))

# Safety caps
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "1024"))

# Generation parameter multipliers and thresholds
# Max tokens multiplier
MAX_TOKENS_MULTIPLIER = float(
    os.environ.get("MAX_TOKENS_MULTIPLIER", "1.0"))
# Minimum tokens to generate
MIN_NEW_TOKENS = int(os.environ.get("MIN_NEW_TOKENS", "20"))
# Repetition penalty (higher = less repetition, reduces hallucination)
REPETITION_PENALTY = float(
    os.environ.get("REPETITION_PENALTY", "1.0"))
# Top-p cap (more conservative = better accuracy)
TOP_P_CAP = float(os.environ.get("TOP_P_CAP", "0.9"))
# Top-k values (more conservative = better accuracy, prevents low-probability tokens)
TOP_K = int(os.environ.get("TOP_K", "20"))

# Explicit device map to avoid offload/meta device warnings.
# Defaults to a single device: "cuda:0" if available else "cpu"
# For CPU-only systems, use "cpu" explicitly
_torch_module = __import__("torch")
DEFAULT_DEVICE = "cuda:0" if _torch_module.cuda.is_available() else "cpu"
DEVICE_MAP = os.environ.get("DEVICE_MAP", DEFAULT_DEVICE)

# RAG (Retrieval-Augmented Generation) settings
# Knowledge base directory - contains subject-specific knowledge files
RAG_KNOWLEDGE_BASE_DIR = os.environ.get(
    "RAG_KNOWLEDGE_BASE_DIR", os.path.join(str(my_path), "knowledge_bases"))
# Number of top-k chunks to retrieve for RAG
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
# Enable RAG for these subjects
RAG_ENABLED_SUBJECTS = {}
