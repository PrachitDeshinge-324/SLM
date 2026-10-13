"""Load a model and its tokenizer at a given weight precision: 16-bit float, or 8-bit / 4-bit bitsandbytes."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PRECISIONS = ("16bit", "8bit", "4bit")

# Model families known to crash on MPS due to PyTorch MPS double-free bugs
# (e.g., Qwen3.5 hybrid GatedDeltaNet + sparse attention architecture)
_MPS_INCOMPATIBLE_FAMILIES = ["qwen3.5", "qwen3-5"]


def select_device(model_id: str = ""):
    """Detects available device: cuda, mps, or cpu.

    MPS is skipped for model families with known MPS memory bugs.
    Set FORCE_CPU=1 environment variable to always use CPU.
    """
    if os.environ.get("FORCE_CPU", "0") == "1":
        print("FORCE_CPU=1 set. Using CPU.")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        model_lower = model_id.lower()
        for family in _MPS_INCOMPATIBLE_FAMILIES:
            if family in model_lower:
                print(f"Warning: '{model_id}' has a known PyTorch MPS double-free bug. Falling back to CPU.")
                print("         (Set FORCE_MPS=1 to override this safety check.)")
                if os.environ.get("FORCE_MPS", "0") != "1":
                    return "cpu"
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, precision: str = "16bit", adapter_path=None):
    """
    Loads a model and tokenizer based on the requested precision.
    precision can be '16bit', '8bit', or '4bit'.

    adapter_path: optional folder of a LoRA adapter trained on this model. It is applied on top of
    the base weights and not merged into them, so a quantized base stays quantized.

    Note: 8-bit and 4-bit require bitsandbytes and typically a CUDA backend.
    """
    device = select_device(model_id)
    print(f"Loading {model_id} on {device} with {precision} precision...")

    if precision in {"8bit", "4bit"} and device != "cuda":
        raise RuntimeError(
            f"{precision} quantization requires CUDA; selected device is {device}."
        )

    # Dynamically check for bfloat16 support to handle a mix of T4, L4, and A100 GPUs
    if device == "cuda" and torch.cuda.is_bf16_supported():
        best_dtype = torch.bfloat16
    elif device == "cpu":
        best_dtype = torch.float32
    else:
        best_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa", # Significantly faster attention for PyTorch 2.0+
    }

    # device_map="auto" for CUDA (handles sharding); explicit device for mps/cpu
    if device == "cuda":
        # Colab normally has one GPU; keep the model there instead of allowing
        # automatic CPU offload, which causes a severe per-token slowdown.
        model_kwargs["device_map"] = "cuda" if torch.cuda.device_count() == 1 else "auto"
    else:
        model_kwargs["device_map"] = device

    if precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=best_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["dtype"] = best_dtype  # For unquantized layers (LM head, embeds)
    elif precision == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["dtype"] = torch.float16  # Native float16 avoids MatMul8bitLt casting warning
    else:
        # 16bit
        model_kwargs["dtype"] = best_dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except (TypeError, ValueError, ImportError) as error:
        if "attn_implementation" not in model_kwargs:
            raise
        print(f"SDPA is unavailable for {model_id}; retrying with the model default attention: {error}")
        model_kwargs.pop("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"Applied LoRA adapter from {adapter_path} (not merged into the {precision} base).")
    model.eval()
    device_map = getattr(model, "hf_device_map", None)
    if device_map and any(str(mapped_device) in {"cpu", "disk"} for mapped_device in device_map.values()):
        print(f"Warning: model placement includes CPU/disk offload: {device_map}")
    elif device_map:
        print(f"Model device map: {device_map}")

    return model, tokenizer


def model_device(model):
    """
    Safely determine the device for a model that may use device_map="auto".
    model.device can fail or return 'meta' for sharded models, so we fall
    back to inspecting the first parameter.
    """
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"
