import sys, json, re, collections, numpy as np
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.experiment_d.answer_extraction import extract_gsm8k_answer
from src.experiment_d.answer_matching import answers_match
from common import *
G = groups()
print("=== (g2) does the agreement shift track per-sample accuracy shift better than majority accuracy shift?")
X = []
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P: continue
    for p in ("8bit", "4bit"):
        if p not in P or not comparable(P, p): continue
        common = sorted(set(P["16bit"]["Q"]) & set(P[p]["Q"]))
        if len(common) < 100: continue
        a = [qstats(P[p]["Q"][q]) for q in common]; b = [qstats(P["16bit"]["Q"][q]) for q in common]
        f = lambda k: np.mean([x[k] for x in a]) - np.mean([x[k] for x in b])
        X.append((d, f("correct"), f("conf_all"), f("single")))
for name, sel in [("all cells", lambda d: True), ("gsm8k", lambda d: d == "gsm8k"), ("cqa", lambda d: d == "cqa")]:
    A = np.array([x[1:] for x in X if sel(x[0])])
    print(f"  {name:10s} n={len(A):2d} target=majority acc: r={np.corrcoef(A[:,0],A[:,1])[0,1]:.2f} MAE={100*np.abs(A[:,0]-A[:,1]).mean():.1f} (no-change {100*np.abs(A[:,0]).mean():.1f}) | target=per-sample acc: r={np.corrcoef(A[:,2],A[:,1])[0,1]:.2f} MAE={100*np.abs(A[:,2]-A[:,1]).mean():.1f} (no-change {100*np.abs(A[:,2]).mean():.1f})")
A = np.array([x[1:] for x in X]); k = np.abs(A[:,2]) < 0.05
print(f"  cells with |dSingle|<5pt (n={k.sum()}): r(dSingle,dConf)={np.corrcoef(A[k,2],A[k,1])[0,1]:.2f} MAE={100*np.abs(A[k,2]-A[k,1]).mean():.1f} no-change={100*np.abs(A[k,2]).mean():.1f}")

print("\n=== (i) is the fallback ('last = N') extraction itself a source of error? Alternative: last number in the final non-empty line.")
NUM = r'-?\d[\d,]*(?:\.\d+)?'
def alt(text):
    t = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.S).strip()
    lines = [l for l in t.split("\n") if re.search(r'\d', l)]
    if not lines: return None
    last = re.sub(r"Let me know.*", "", lines[-1])
    nums = re.findall(NUM, last.replace("\\$", "").replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "gsm8k", "generator", "%s", "%s_%s_gsm8k_n8.jsonl")
for model in ["google_gemma-3-1b-it", "meta-llama_Llama-3.2-1B-Instruct", "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B"]:
    for prec in PREC:
        n = same = acc_cur = acc_alt = either = 0
        for line in open(base % (model, model, prec)):
            r = json.loads(line)
            if "raw_samples" not in r: continue
            for s in r["raw_samples"]:
                if s["cutoff"]: continue
                t = s["response"]
                if "####" in t or "\\boxed{" in t: continue
                cur, al = extract_gsm8k_answer(t), alt(t)
                c1, c2 = answers_match(cur, r["ground_truth"]), answers_match(al, r["ground_truth"])
                n += 1; same += (cur is not None and al is not None and answers_match(cur, al)); acc_cur += c1; acc_alt += c2; either += (c1 or c2)
        if n: print(f"  {short(model):13s} {prec:5s} fallback-path samples {n:5d}: current extractor acc {acc_cur/n:.3f} | alt extractor acc {acc_alt/n:.3f} | two extractors agree {same/n:.3f} | correct under either {either/n:.3f}")
