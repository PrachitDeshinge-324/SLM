"""
Does vLLM give the same verdicts as Hugging Face? A small parity check of the one-line verifier (v0).

Takes the first --questions questions of a stored v0 verifier run (all their solutions), rebuilds exactly
the prompts the run used (src/experiment_d/verifier.py, the model's chat template, no special tokens added),
and asks the same model again, greedy, through each engine in --engines:
  hf    Hugging Face generate, as run_verifier.py does (dtype --dtype)
  vllm  vLLM, temperature 0, fed the same prompt token ids (dtype --dtype; a T4 has no bfloat16 in vLLM)
Each is compared with the stored verdicts and with the other: agreement, Cohen's kappa, which verdicts flip,
and the verifier's approval rates on right and wrong solutions (the reference answers are used only there).
Reading: stored vs hf shows the effect of dtype and hardware alone; hf vs vllm, at the same dtype, the engine.

Output: results/<dataset>/engine_parity/<model>/v0_<precision>.json with every verdict and the summary.
Needs a CUDA GPU (vllm also needs `pip install vllm`); engines run one after the other.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.experiment_d.warning_filters import silence_harmless_warnings

silence_harmless_warnings()
load_dotenv()
sys.stdout.reconfigure(line_buffering=True)

from src.experiment_d.engine_parity import agreement, verifier_rates
from src.experiment_d.prompts import build_prompt
from src.experiment_d.rescoring import relabel_trace
from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.run_files import parse_n_samples, safe_name, verifier_output_path
from src.experiment_d.sampling import count_generated_tokens
from src.experiment_d.verifier import get_verifier_messages, parse_verdict


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM against Hugging Face on the one-line verifier's verdicts")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=("16bit", "4bit"), default="16bit")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--n_samples", type=str, default="8")
    parser.add_argument("--questions", type=int, default=25, help="First questions of the stored run, all their solutions")
    parser.add_argument("--engines", type=str, default="hf,vllm")
    parser.add_argument("--dtype", type=str, choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="The stored run allowed 2048; a one-line verdict needs far fewer (cut-offs are reported)")
    parser.add_argument("--batch_size", type=int, default=16, help="Hugging Face batch size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    args = parser.parse_args()
    args.n_samples = parse_n_samples(args.n_samples)
    return args


def stored_traces(args) -> list:
    path = verifier_output_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples)
    if not path.exists():
        raise FileNotFoundError(f"No stored v0 verifier run at {path}")
    extractor, records, questions = get_extractor(args.dataset), [], []
    for line in path.open():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["question"] not in questions:
            if len(questions) == args.questions:
                break
            questions.append(record["question"])
        records.append(record)
    print(f"{path.name}: first {len(questions)} questions, {len(records)} solutions.")
    return [{"question": r["question"], "solution": r["student_solution"],
             "stored_response": r.get("verifier_raw_response"),
             "stored": parse_verdict(r["verifier_raw_response"]) if "verifier_raw_response" in r else r.get("verifier_final_conclusion"),
             "correct": relabel_trace(r, extractor)["correct"]} for r in records]


def run_hf(args, tokenizer_prompts: list) -> tuple:
    import torch
    from src.experiment_d.model_loading import load_model_and_tokenizer, model_device
    model, tokenizer = load_model_and_tokenizer(args.model, args.precision, dtype=args.dtype)
    tokenizer.padding_side = "left"
    out = []
    for start in range(0, len(tokenizer_prompts), args.batch_size):
        batch = tokenizer_prompts[start:start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model_device(model))
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                       repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
        new = generated[:, inputs["input_ids"].shape[1]:]
        out += [{"text": t, "tokens": count_generated_tokens(row, tokenizer)}
                for t, row in zip(tokenizer.batch_decode(new, skip_special_tokens=True), new)]
    info = {"engine": "hf", "dtype": str(model.dtype)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, info


def run_vllm(args, tokenizer, prompts: list) -> tuple:
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    kwargs = {"model": args.model, "dtype": args.dtype, "max_model_len": 4096, "seed": 0, "trust_remote_code": True,
              "gpu_memory_utilization": args.gpu_memory_utilization, "limit_mm_per_prompt": {"image": 0, "video": 0}}
    if args.precision == "4bit":
        kwargs.update(quantization="bitsandbytes", load_format="bitsandbytes")
    llm = vllm.LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, repetition_penalty=1.0,
                            skip_special_tokens=True)
    ids = [TokensPrompt(prompt_token_ids=tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    results = llm.generate(ids, params)
    out = [{"text": r.outputs[0].text, "tokens": len(r.outputs[0].token_ids)} for r in results]
    info = {"engine": "vllm", "dtype": args.dtype, "vllm_version": vllm.__version__}
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return out, info


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    traces = stored_traces(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [build_prompt(tokenizer, get_verifier_messages(t["question"], t["solution"])) for t in traces]
    correct = [t["correct"] for t in traces]
    verdicts = {"stored": [t["stored"] for t in traces]}
    engines, outputs = {}, {}
    for engine in [e for e in args.engines.split(",") if e]:
        start = time.time()
        out, info = run_hf(args, prompts) if engine == "hf" else run_vllm(args, tokenizer, prompts)
        info["seconds"] = round(time.time() - start, 1)
        info["cut_off"] = sum(o["tokens"] >= args.max_new_tokens for o in out)
        engines[engine], outputs[engine] = info, out
        verdicts[engine] = [parse_verdict(o["text"]) for o in out]
        print(f"{engine}: {info}")

    summary = {"rates": {name: verifier_rates(v, correct) for name, v in verdicts.items()}, "pairs": {}}
    names = list(verdicts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            summary["pairs"][f"{a} vs {b}"] = agreement(verdicts[a], verdicts[b])
    print(f"\n{len(traces)} solutions ({sum(correct)} right). Verdicts per engine:")
    print("engine | parsed | approved | approved when right (TPR) | approved when wrong (FPR)")
    fmt = lambda x: "-" if x is None else f"{x:.3f}"
    for name, r in summary["rates"].items():
        print(f"{name} | {fmt(r['parsed'])} | {fmt(r['approved'])} | {fmt(r['tpr'])} | {fmt(r['fpr'])}")
    print("\npair | agreement | kappa | flips")
    for pair, r in summary["pairs"].items():
        print(f"{pair} | {r['agreement']:.3f} | {r['kappa']:.3f} | {r['flips']}")

    out_path = Path(args.results_dir) / args.dataset / "engine_parity" / safe_name(args.model) / f"v0_{args.precision}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"settings": {k: v for k, v in vars(args).items()}, "engines": engines,
                                    "summary": summary,
                                    "traces": [{**t, **{f"{e}_response": outputs[e][i]["text"] for e in outputs},
                                                **{f"{e}_verdict": verdicts[e][i] for e in outputs}}
                                               for i, t in enumerate(traces)]}, indent=1))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
