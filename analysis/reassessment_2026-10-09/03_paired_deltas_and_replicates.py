from common import *
G = groups()
def pq(Q, q, idx=None):
    S = Q[q]["S"] if idx is None else [Q[q]["S"][i] for i in idx]
    return np.mean([s["correct"] and not s["cut"] for s in S])

print("=== (a) paired deltas vs 16-bit, cluster bootstrap over questions: single-sample accuracy and majority accuracy")
print("cell | prec | nQ | d_single [95% CI] | d_majority [95% CI] | d_pass@k [CI]")
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P: continue
    for p in ("8bit", "4bit"):
        if p not in P: continue
        common = sorted(set(P["16bit"]["Q"]) & set(P[p]["Q"]))
        if len(common) < 50: continue
        flag = "" if comparable(P, p) else "  <-- token limits differ, not a valid contrast"
        a = np.array([pq(P[p]["Q"], q) for q in common]); b = np.array([pq(P["16bit"]["Q"], q) for q in common])
        ya = np.array([qstats(P[p]["Q"][q])["correct"] for q in common], float); yb = np.array([qstats(P["16bit"]["Q"][q])["correct"] for q in common], float)
        pa = np.array([qstats(P[p]["Q"][q])["pass"] for q in common], float); pb = np.array([qstats(P["16bit"]["Q"][q])["pass"] for q in common], float)
        N = len(common)
        c1 = bootstrap_interval(lambda ix: (a[ix] - b[ix]).mean(), N); c2 = bootstrap_interval(lambda ix: (ya[ix] - yb[ix]).mean(), N); c3 = bootstrap_interval(lambda ix: (pa[ix] - pb[ix]).mean(), N)
        star = lambda c: "*" if c[0] > 0 or c[1] < 0 else " "
        print(f"{d[:7]:7s} {short(m):13s} n{n:<2d} | {p:4s} | {N:4d} | {100*(a-b).mean():+5.1f} [{100*c1[0]:+5.1f},{100*c1[1]:+5.1f}]{star(c1)} | {100*(ya-yb).mean():+5.1f} [{100*c2[0]:+5.1f},{100*c2[1]:+5.1f}]{star(c2)} | {100*(pa-pb).mean():+5.1f} [{100*c3[0]:+5.1f},{100*c3[1]:+5.1f}]{star(c3)}{flag}")

print("\n=== (b) replicate noise: same model+precision, n8 run vs first 8 samples of the n16 run (independent runs)")
print("cell | prec | nQ | identical-response share | d_single (n16first8 - n8) [CI] | d_majority [CI] | majority-answer disagreement | correctness flip")
for (d, m, n), P in sorted(G.items()):
    if n != 8 or (d, m, 16) not in G: continue
    for p in PREC:
        if p not in P or p not in G[(d, m, 16)]: continue
        A, B = P[p]["Q"], G[(d, m, 16)][p]["Q"]
        common = sorted(set(A) & set(B)); N = len(common)
        same = np.mean([len({s["h"] for s in A[q]["S"]} & {s["h"] for s in B[q]["S"][:8]}) > 0 for q in common])
        a = np.array([pq(A, q) for q in common]); b = np.array([pq(B, q, range(8)) for q in common])
        sa = [qstats(A[q]) for q in common]; sb = [qstats(B[q], range(8)) for q in common]
        ya = np.array([s["correct"] for s in sa], float); yb = np.array([s["correct"] for s in sb], float)
        dis = np.mean([x["maj"] != y["maj"] for x, y in zip(sa, sb)])
        c1 = bootstrap_interval(lambda ix: (b[ix] - a[ix]).mean(), N); c2 = bootstrap_interval(lambda ix: (yb[ix] - ya[ix]).mean(), N)
        print(f"{d[:7]:7s} {short(m):13s} | {p:5s} | {N:4d} | {same:.3f} | {100*(b-a).mean():+5.1f} [{100*c1[0]:+5.1f},{100*c1[1]:+5.1f}] | {100*(yb-ya).mean():+5.1f} [{100*c2[0]:+5.1f},{100*c2[1]:+5.1f}] | {dis:.3f} | {np.mean(ya!=yb):.3f}")
