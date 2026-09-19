import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import time


def get_device():
    """Detects available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, precision: str = "16bit"):
    """
    Loads a model and tokenizer based on the requested precision.
    precision can be '16bit', '8bit', or '4bit'.

    Note: 8-bit and 4-bit require bitsandbytes and typically a CUDA backend.
    """
    device = get_device()
    print(f"Loading {model_id} on {device} with {precision} precision...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa", # Significantly faster attention for PyTorch 2.0+
    }

    # device_map="auto" for CUDA (handles sharding); explicit device for mps/cpu
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = device

    if precision == "4bit":
        if device != "cuda":
            print("Warning: 4-bit quantization usually requires CUDA. Attempting anyway...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16  # For unquantized layers (LM head, embeds)
    elif precision == "8bit":
        if device != "cuda":
            print("Warning: 8-bit quantization usually requires CUDA. Attempting anyway...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.float16  # Native float16 avoids MatMul8bitLt casting warning
    else:
        # 16bit / bfloat16
        if device == "mps":
            model_kwargs["torch_dtype"] = torch.float16  # mps prefers float16
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    return model, tokenizer


def _get_model_device(model):
    """
    Safely determine the device for a model that may use device_map="auto".
    model.device can fail or return 'meta' for sharded models, so we fall
    back to inspecting the first parameter.
    """
    try:
        dev = model.device
        if dev is not None and str(dev) != "meta":
            return dev
    except Exception:
        pass
    return next(model.parameters()).device


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
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    all_responses = []
    generation_lengths = []
    total_new_tokens = 0
    start_time = time.time()

    remaining = n_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)

        input_ids = inputs["input_ids"].repeat(current_batch, 1)
        attention_mask = inputs["attention_mask"].repeat(current_batch, 1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Slice to only the newly generated tokens
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]
        
        # Calculate length of each generated sequence
        for i in range(current_batch):
            seq = generated_tokens[i]
            valid_len = (seq != tokenizer.pad_token_id).sum().item()
            generation_lengths.append(valid_len)

        total_new_tokens += generated_tokens.numel()

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
):
    """
    Generates n_samples for multiple prompts at once to maximize GPU utilization.
    Returns a list of lists of decoded responses: [[resp1_for_prompt1, ...], [resp1_for_prompt2, ...]]
    """
    device = _get_model_device(model)
    
    # Ensure left padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    # Repeat each prompt n_samples times
    # If prompts=[P1, P2], n_samples=2 -> [P1, P1, P2, P2]
    input_ids = torch.repeat_interleave(inputs["input_ids"], repeats=n_samples, dim=0)
    attention_mask = torch.repeat_interleave(inputs["attention_mask"], repeats=n_samples, dim=0)
    
    total_sequences = input_ids.shape[0]
    
    all_responses_flat = []
    generation_lengths_flat = []
    
    total_new_tokens = 0
    start_time = time.time()
    
    for i in range(0, total_sequences, batch_size):
        end = min(i + batch_size, total_sequences)
        
        batch_input_ids = input_ids[i:end]
        batch_attention_mask = attention_mask[i:end]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        prompt_length = batch_input_ids.shape[1]
        generated_tokens = outputs[:, prompt_length:]
        
        for j in range(generated_tokens.shape[0]):
            seq = generated_tokens[j]
            valid_len = (seq != tokenizer.pad_token_id).sum().item()
            generation_lengths_flat.append(valid_len)
            
        total_new_tokens += generated_tokens.numel()
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_responses_flat.extend(decoded)
        
    end_time = time.time()
    latency = end_time - start_time
    tokens_per_sec = total_new_tokens / latency if latency > 0 else 0
    
    # Restore padding side
    tokenizer.padding_side = original_padding_side
    
    # Group responses back by prompt
    grouped_responses = []
    grouped_lengths = []
    
    for i in range(len(prompts)):
        start = i * n_samples
        end = start + n_samples
        grouped_responses.append(all_responses_flat[start:end])
        grouped_lengths.append(generation_lengths_flat[start:end])
        
    return grouped_responses, tokens_per_sec, latency, grouped_lengths
