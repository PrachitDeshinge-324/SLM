import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)
import time


import os

# Model families known to crash on MPS due to PyTorch MPS double-free bugs
# (e.g., Qwen3.5 hybrid GatedDeltaNet + sparse attention architecture)
_MPS_INCOMPATIBLE_FAMILIES = ["qwen3.5", "qwen3-5"]


class _StopAfterBoxedAnswer(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        should_stop = []
        for sequence in input_ids:
            generated = sequence[self.prompt_length:][-256:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            start = text.rfind("\\boxed{")
            if start < 0:
                should_stop.append(False)
                continue

            depth = 0
            for character in text[start + len("\\boxed{") - 1:]:
                if character == "{" and (depth == 0 or character != "\\"):
                    depth += 1
                elif character == "}" and depth > 0:
                    depth -= 1
                    if depth == 0:
                        break
            should_stop.append(depth == 0)
        return torch.tensor(should_stop, dtype=torch.bool, device=input_ids.device)


def get_device(model_id: str = ""):
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


def load_model_and_tokenizer(model_id: str, precision: str = "16bit"):
    """
    Loads a model and tokenizer based on the requested precision.
    precision can be '16bit', '8bit', or '4bit'.

    Note: 8-bit and 4-bit require bitsandbytes and typically a CUDA backend.
    """
    device = get_device(model_id)
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
    model.eval()
    device_map = getattr(model, "hf_device_map", None)
    if device_map and any(str(mapped_device) in {"cpu", "disk"} for mapped_device in device_map.values()):
        print(f"Warning: model placement includes CPU/disk offload: {device_map}")
    elif device_map:
        print(f"Model device map: {device_map}")

    return model, tokenizer


def _get_model_device(model):
    """
    Safely determine the device for a model that may use device_map="auto".
    model.device can fail or return 'meta' for sharded models, so we fall
    back to inspecting the first parameter.
    """
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _generated_token_length(tokens, tokenizer):
    """Return generated tokens before the first EOS or padding token."""
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    for index, token in enumerate(tokens.tolist()):
        if token == eos_token_id or token == pad_token_id:
            return index
    return len(tokens)


def generate_n_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 16,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    repetition_penalty: float = 1.0,
    stop_on_boxed: bool = False,
):
    """
    Generates N samples for a given prompt using temperature sampling.
    Processes in mini-batches of `batch_size` to avoid OOM.

    Returns:
        decoded_responses: list[str] of N generated strings.
        tokens_per_sec: float, overall generation throughput.
        latency: float, total wall-clock seconds.
        generation_lengths: list[int] of lengths of newly generated tokens.
    """
    device = _get_model_device(model)
    # The chat template already writes BOS; do not let the tokenizer add a second one.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    all_responses = []
    generation_lengths = []
    total_new_tokens = 0
    start_time = time.time()

    remaining = n_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)

        input_ids = inputs["input_ids"].repeat(current_batch, 1)
        attention_mask = inputs["attention_mask"].repeat(current_batch, 1)

        stopping_criteria = (
            StoppingCriteriaList([_StopAfterBoxedAnswer(tokenizer, inputs["input_ids"].shape[1])])
            if stop_on_boxed else None
        )
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Slice to only the newly generated tokens
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]
        
        # Calculate length of each generated sequence
        batch_valid_tokens = 0
        for i in range(current_batch):
            seq = generated_tokens[i]
            valid_len = _generated_token_length(seq, tokenizer)
            generation_lengths.append(valid_len)
            batch_valid_tokens += valid_len

        total_new_tokens += batch_valid_tokens

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_responses.extend(decoded)

        remaining -= current_batch

    end_time = time.time()
    latency = end_time - start_time
    tokens_per_sec = total_new_tokens / latency if latency > 0 else 0

    return all_responses, tokens_per_sec, latency, generation_lengths

def generate_batch_prompts(
    model,
    tokenizer,
    prompts: list,
    n_samples: int = 16,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    batch_size: int = 32,
    repetition_penalty: float = 1.0,
    stop_on_boxed: bool = False,
):
    """
    Generates n_samples for multiple prompts at once to maximize GPU utilization.
    Returns a list of lists of decoded responses: [[resp1_for_prompt1, ...], [resp1_for_prompt2, ...]]
    """
    device = _get_model_device(model)
    
    # Ensure left padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    try:
        # The chat template already writes BOS; do not let the tokenizer add a second one.
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        
        # Expand prompts lazily per generation mini-batch. Materializing every
        # repeated prompt up front can waste substantial VRAM at large batch sizes.
        total_sequences = len(prompts) * n_samples
        
        all_responses_flat = []
        generation_lengths_flat = []
        
        total_new_tokens = 0
        start_time = time.time()
        
        for i in range(0, total_sequences, batch_size):
            end = min(i + batch_size, total_sequences)
            
            prompt_indices = torch.arange(i, end, device=inputs["input_ids"].device) // n_samples
            batch_input_ids = inputs["input_ids"].index_select(0, prompt_indices)
            batch_attention_mask = inputs["attention_mask"].index_select(0, prompt_indices)
            
            stopping_criteria = (
                StoppingCriteriaList([_StopAfterBoxedAnswer(tokenizer, batch_input_ids.shape[1])])
                if stop_on_boxed else None
            )
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=tokenizer.pad_token_id,
                )
                
            prompt_length = batch_input_ids.shape[1]
            generated_tokens = outputs[:, prompt_length:]
            
            batch_valid_tokens = 0
            for j in range(generated_tokens.shape[0]):
                seq = generated_tokens[j]
                valid_len = _generated_token_length(seq, tokenizer)
                generation_lengths_flat.append(valid_len)
                batch_valid_tokens += valid_len
                
            total_new_tokens += batch_valid_tokens
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_responses_flat.extend(decoded)
            
        end_time = time.time()
        latency = end_time - start_time
        tokens_per_sec = total_new_tokens / latency if latency > 0 else 0
        
        # Group responses back by prompt
        grouped_responses = []
        grouped_lengths = []
        
        for i in range(len(prompts)):
            start = i * n_samples
            end = start + n_samples
            grouped_responses.append(all_responses_flat[start:end])
            grouped_lengths.append(generation_lengths_flat[start:end])
    finally:
        # Always restore the caller's padding preference, even on OOM or other errors.
        tokenizer.padding_side = original_padding_side
        
    return grouped_responses, tokens_per_sec, latency, grouped_lengths
