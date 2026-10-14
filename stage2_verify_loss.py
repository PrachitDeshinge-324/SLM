import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Official Hugging Face repo ID
MODEL_ID = "Qwen/Qwen3.5-0.8B"
PAIRS_FILE = "dpo_pairs.jsonl"
BETA = 0.1
CLIP_C = 0.2

# Force CPU on Mac to avoid MPS weight-loading segfaults
DEVICE = "cpu"
print(f"Running Stage 2 verification on: {DEVICE} (safe mode for verification)")

# 1. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading {MODEL_ID} weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    low_cpu_mem_usage=True
).to(DEVICE)

# 2. Attach LoRA Adapter
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.train()
print("Attached LoRA adapter successfully.")

# ==============================================================================
# 3. LENGTH-NORMALIZED CLIPPED DPO LOSS FUNCTION
# ==============================================================================
def get_per_token_logps(model, input_ids, labels):
    """Computes per-token log-probabilities masked to response tokens only."""
    logits = model(input_ids).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())

    # Mask out prompt tokens (-100)
    mask = (shift_labels != -100).float()
    per_token_logps = -(loss * mask)

    # Per-token mean (Length Normalization)
    lengths = mask.sum(dim=-1).clamp(min=1.0)
    mean_logps = per_token_logps.sum(dim=-1) / lengths
    return mean_logps

def compute_dpo_loss(model, prompt, chosen, rejected):
    """Computes Length-Normalized DPO with Dual-Clip."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    chosen_ids = tokenizer(chosen, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    rejected_ids = tokenizer(rejected, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]

    c_input_ids = torch.tensor([prompt_ids + chosen_ids], device=DEVICE)
    c_labels = torch.tensor([[-100] * len(prompt_ids) + chosen_ids], device=DEVICE)

    r_input_ids = torch.tensor([prompt_ids + rejected_ids], device=DEVICE)
    r_labels = torch.tensor([[-100] * len(prompt_ids) + rejected_ids], device=DEVICE)

    # 1. Policy forward pass (adapter enabled)
    policy_chosen_logp = get_per_token_logps(model, c_input_ids, c_labels)
    policy_rejected_logp = get_per_token_logps(model, r_input_ids, r_labels)

    # 2. Reference forward pass (adapter disabled)
    with torch.no_grad():
        with model.disable_adapter():
            ref_chosen_logp = get_per_token_logps(model, c_input_ids, c_labels)
            ref_rejected_logp = get_per_token_logps(model, r_input_ids, r_labels)

    # Delta (implicit reward difference)
    delta = (policy_chosen_logp - ref_chosen_logp) - (policy_rejected_logp - ref_rejected_logp)

    # Dual-Clip objective
    clipped_delta = torch.clamp(delta, -CLIP_C, CLIP_C)
    loss = -F.logsigmoid(BETA * clipped_delta).mean()

    return loss, delta.item()

# ==============================================================================
# 4. VERIFY ON 2 PAIRS (FAST AUDIT)
# ==============================================================================
print(f"Loading sample pairs from {PAIRS_FILE}...")
test_pairs = []
with open(PAIRS_FILE, "r") as f:
    for line in f:
        if line.strip():
            test_pairs.append(json.loads(line))
        if len(test_pairs) == 2:
            break

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("\n" + "=" * 50)
print("STAGE 2 GRADIENT & LOSS AUDIT")
print("=" * 50)

for i, pair in enumerate(test_pairs):
    optimizer.zero_grad()
    loss, delta = compute_dpo_loss(model, pair["prompt"], pair["chosen"], pair["rejected"])
    loss.backward()

    # Calculate gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    optimizer.step()

    print(f"Pair {i+1}:")
    print(f"  Loss:               {loss.item():.4f}")
    print(f"  Implicit Reward Δ:  {delta:+.4f}")
    print(f"  Total Grad Norm:    {total_norm:.4f}")
    assert not torch.isnan(loss), "Fatal: Loss is NaN!"
    assert total_norm < 10.0, "Fatal: Gradient explosion!"

print("\n[SUCCESS] Stage 2 passed: Loss is stable, gradients are bounded, and length-normalized backprop works.")
