from common import *
import collections
print("=== extraction path x precision (complete samples only): share of samples, accuracy within path")
for (d, m, n), P in sorted(groups().items()):
    if d not in ("gsm8k", "ChilleD_SVAMP", "math500") or n != 8: continue
    common = sorted(set.intersection(*[set(v["Q"]) for v in P.values()]))
    if len(common) < 20: continue
    print(f"\n{d} {short(m)} nQ={len(common)}")
    for p in PREC:
        if p not in P: continue
        c = collections.Counter(); k = collections.Counter(); tot = 0
        for q in common:
            for s in P[p]["Q"][q]["S"]:
                if s["cut"]: continue
                c[s["path"]] += 1; k[s["path"]] += s["correct"]; tot += 1
        print(f"  {p:5s} " + "  ".join(f"{path}:{c[path]/tot:.2f}(acc {k[path]/c[path]:.2f})" for path, _ in c.most_common()))
