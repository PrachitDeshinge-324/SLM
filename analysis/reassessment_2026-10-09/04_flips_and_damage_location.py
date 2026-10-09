from common import *
G = groups()
def pq(Q, q, idx=None):
    S = Q[q]["S"] if idx is None else [Q[q]["S"][i] for i in idx]
    return np.mean([s["correct"] and not s["cut"] for s in S])
def yq(Q, q, idx=None): return qstats(Q[q], idx)["correct"]
def mq(Q, q, idx=None): return qstats(Q[q], idx)["maj"]

print("=== (c) correctness flips and majority-answer changes: cross-precision vs same-precision replicate floor (all at 8 samples)")
for d, m in [("gsm8k", "Qwen_Qwen3.5-0.8B"), ("cqa", "Qwen_Qwen3.5-0.8B")]:
    A, B = G[(d, m, 8)], G[(d, m, 16)]
    common = sorted(set.intersection(*[set(v["Q"]) for v in list(A.values()) + list(B.values())]))
    N = len(common); f8 = range(8)
    def flips(Q1, i1, Q2, i2):
        y1 = np.array([yq(Q1, q, i1) for q in common]); y2 = np.array([yq(Q2, q, i2) for q in common])
        m1 = [mq(Q1, q, i1) for q in common]; m2 = [mq(Q2, q, i2) for q in common]
        return np.mean(y1 != y2), np.mean((y1 == 1) & (y2 == 0)), np.mean((y1 == 0) & (y2 == 1)), np.mean([a != b for a, b in zip(m1, m2)])
    print(f"\n{d} {short(m)} nQ={N}   [flip total | correct->wrong | wrong->correct | majority answer differs]")
    for p in PREC:
        print(f"  floor  {p:5s} n8-run vs n16-run(first 8):   " + "  ".join(f"{x:.3f}" for x in flips(A[p]["Q"], None, B[p]["Q"], f8)))
    for p in ("8bit", "4bit"):
        r = [flips(A["16bit"]["Q"], None, A[p]["Q"], None), flips(B["16bit"]["Q"], f8, B[p]["Q"], f8), flips(A["16bit"]["Q"], None, B[p]["Q"], f8), flips(B["16bit"]["Q"], f8, A[p]["Q"], None)]
        print(f"  cross  16bit->{p:4s} (mean of 4 run pairings): " + "  ".join(f"{np.mean([x[i] for x in r]):.3f}" for i in range(4)) + "   range of flip total: " + f"{min(x[0] for x in r):.3f}-{max(x[0] for x in r):.3f}")

print("\n=== (d) where does 4-bit damage land? Questions binned by 16-bit per-sample accuracy from an INDEPENDENT run (n16 run, 16 samples); delta measured on the n8 runs")
for d, m in [("gsm8k", "Qwen_Qwen3.5-0.8B"), ("cqa", "Qwen_Qwen3.5-0.8B")]:
    A, B = G[(d, m, 8)], G[(d, m, 16)]
    common = sorted(set.intersection(*[set(v["Q"]) for v in list(A.values()) + list(B.values())]))
    ref = np.array([pq(B["16bit"]["Q"], q) for q in common])
    p16 = np.array([pq(A["16bit"]["Q"], q) for q in common]); p8 = np.array([pq(A["8bit"]["Q"], q) for q in common]); p4 = np.array([pq(A["4bit"]["Q"], q) for q in common])
    r16 = np.array([pq(B["16bit"]["Q"], q, range(8)) for q in common])  # not independent of ref; shown only for the floor row below
    print(f"\n{d} {short(m)}: bin by independent 16-bit p | nQ | p16 (n8 run) | p8 | p4 | d8 | d4 | share of total 4-bit loss")
    tot = (p4 - p16).sum()
    for lo, hi in [(0, 0.001), (0.001, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.01)]:
        k = (ref >= lo) & (ref < hi)
        if k.sum() == 0: continue
        print(f"  [{lo:.2f},{min(hi,1):.2f}) | {k.sum():4d} | {p16[k].mean():.3f} | {p8[k].mean():.3f} | {p4[k].mean():.3f} | {100*(p8[k]-p16[k]).mean():+5.1f} | {100*(p4[k]-p16[k]).mean():+5.1f} | {100*(p4[k]-p16[k]).sum()/tot:5.1f}%")

print("\n=== (e) per-question shift with 24 samples per precision (n8+n16 runs pooled), Qwen-0.8B GSM8K")
A, B = G[("gsm8k", "Qwen_Qwen3.5-0.8B", 8)], G[("gsm8k", "Qwen_Qwen3.5-0.8B", 16)]
common = sorted(set.intersection(*[set(v["Q"]) for v in list(A.values()) + list(B.values())]))
def pool(p):
    return np.array([np.mean([s["correct"] and not s["cut"] for s in A[p]["Q"][q]["S"] + B[p]["Q"][q]["S"]]) for q in common])
def half(p, which):
    out = []
    for q in common:
        S = A[p]["Q"][q]["S"] + B[p]["Q"][q]["S"]
        idx = [i for i in range(24) if (i % 2) == which]
        out.append(np.mean([S[i]["correct"] and not S[i]["cut"] for i in idx]))
    return np.array(out)
P16, P8, P4 = pool("16bit"), pool("8bit"), pool("4bit")
null = half("16bit", 0) - half("16bit", 1)          # 12 vs 12 samples, same precision
d4h = half("4bit", 0) - half("16bit", 1); d8h = half("8bit", 0) - half("16bit", 1)   # 12 vs 12, cross precision
for name, x in [("null 16 vs 16 (12 v 12)", null), ("8bit vs 16bit (12 v 12)", d8h), ("4bit vs 16bit (12 v 12)", d4h)]:
    print(f"  {name:26s} mean {100*x.mean():+5.1f}  sd {x.std():.3f}  P(d<=-0.5) {np.mean(x<=-0.5):.3f}  P(d>=+0.5) {np.mean(x>=0.5):.3f}  P(|d|<0.17) {np.mean(np.abs(x)<0.17):.3f}")
solv16 = P16 > 0; print(f"  questions with >=1 correct of 24: 16bit {np.mean(P16>0):.3f}  8bit {np.mean(P8>0):.3f}  4bit {np.mean(P4>0):.3f};  solvable at 16 but 0/24 at 4bit: {np.mean((P16>0)&(P4==0)):.3f}; reverse: {np.mean((P16==0)&(P4>0)):.3f};  16bit>=0.5 & 4bit==0: {np.mean((P16>=0.5)&(P4==0)):.3f}")
