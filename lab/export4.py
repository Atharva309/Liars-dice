import json
out = {"agents": {}, "curves": {}}
for n in ("cfr", "dqn", "ppo", "nfsp", "mc"):
    d = json.load(open(f"out4/{n}.json"))
    pol = d["policy"]
    if d["kind"] == "table": pol = {k: [[a, round(p, 2)] for a, p in v if p >= 0.01] for k, v in pol.items()}
    out["agents"][n] = {"kind": d["kind"], "policy": pol}; out["curves"][n] = d["curve"]
json.dump(out, open("site/agents4.json", "w"), separators=(",", ":"))
import os; print(os.path.getsize("site/agents4.json") / 1e6, "MB")
