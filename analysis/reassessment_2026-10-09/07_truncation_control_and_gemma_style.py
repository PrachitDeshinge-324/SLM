from common import *
G = groups()
print("=== (1) agreement-shift estimator restricted to questions with NO truncated sample at either precision (truncation control); target = per-sample accuracy")
X = []
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P: continue
    for p in ("8bit", "4bit"):
        if p not in P or not comparable(P, p): continue
        common = [q for q in sorted(set(P["16bit"]["Q"]) & set(P[p]["Q"])) if not any(s["cut"] for s in P["16bit"]["Q"][q]["S"]) and not any(s["cut"] for s in P[p]["Q"][q]["S"])]
        if len(common) < 100: continue
        a = [qstats(P[p]["Q"][q]) for q in common]; b = [qstats(P["16bit"]["Q"][q]) for q in common]
        f = lambda k: np.mean([x[k] for x in a]) - np.mean([x[k] for x in b])
        X.append((f("correct"), f("conf_all"), f("single"), len(common), d, short(m), n, p))
A = np.array([x[:3] for x in X])
for x in X: print(f"   {x[4][:6]:6s} {x[5]:13s} n{x[6]:<2d} {x[7]:4s} nQ={x[3]:4d} dMaj {100*x[0]:+5.1f} dSingle {100*x[2]:+5.1f} dConf {100*x[1]:+5.1f}")
print(f"  cells={len(A)}  target majority: r={np.corrcoef(A[:,0],A[:,1])[0,1]:.2f} MAE={100*np.abs(A[:,0]-A[:,1]).mean():.1f} no-change={100*np.abs(A[:,0]).mean():.1f} | target per-sample: r={np.corrcoef(A[:,2],A[:,1])[0,1]:.2f} MAE={100*np.abs(A[:,2]-A[:,1]).mean():.1f} no-change={100*np.abs(A[:,2]).mean():.1f}")

print("\n=== (2) per-sample accuracy among COMPLETE samples only (to compare with published pass@1 numbers that use a 32k token limit)")
for (d, m, n), P in sorted(G.items()):
    if n != 8 or d not in ("gsm8k",): continue
    row = f"  {short(m):13s}"
    for p in PREC:
        if p in P:
            S = [s for q in P[p]["Q"].values() for s in q["S"] if not s["cut"]]
            row += f" {p}: {np.mean([s['correct'] for s in S]):.3f}"
    print(row)

print("\n=== (3) Gemma-1B: is answer style a marker of the question or of the sample? Within-question comparison, questions having both styles among complete samples")
P = G[("gsm8k", "google_gemma-3-1b-it", 8)]
for p in PREC:
    d = []; share = []
    for q in P[p]["Q"].values():
        S = [s for s in q["S"] if not s["cut"]]
        bx = [s["correct"] for s in S if s["path"] == "boxed"]; ot = [s["correct"] for s in S if s["path"] != "boxed"]
        if S: share.append(len(bx) / len(S))
        if bx and ot: d.append(np.mean(bx) - np.mean(ot))
    share = np.array(share)
    print(f"  {p:5s} questions with both styles: {len(d):3d}; within-question acc(boxed) - acc(other) = {np.mean(d):+.3f} (se {np.std(d)/np.sqrt(len(d)):.3f}); questions all-boxed {np.mean(share==1):.2f}, none-boxed {np.mean(share==0):.2f}")
A16 = {q: np.mean([s["path"] == "boxed" for s in v["S"] if not s["cut"]] or [np.nan]) for q, v in P["16bit"]["Q"].items()}
A4 = {q: np.mean([s["path"] == "boxed" for s in v["S"] if not s["cut"]] or [np.nan]) for q, v in P["4bit"]["Q"].items()}
qs = [q for q in A16 if not np.isnan(A16[q]) and not np.isnan(A4[q])]
print(f"  per-question boxed share 16-bit vs 4-bit: r = {np.corrcoef([A16[q] for q in qs],[A4[q] for q in qs])[0,1]:.2f} (n={len(qs)})")
