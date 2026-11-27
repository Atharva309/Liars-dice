import json, random, sys, time, numpy as np
import mp as M
from nn import MLP
from ismcts_mp import make_ismcts
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
def load_net(d):
    n = MLP([1]); n.W = [np.array(w, np.float32) for w in d["W"]]; n.b = [np.array(b, np.float32) for b in d["b"]]; return n
def make(name, seed):
    rng = random.Random(seed)
    if name == "math": return M.math_bot
    if name == "random": return M.make_random(rng)
    if name == "ismcts": return make_ismcts(700, 1.0, seed)
    d = json.load(open(f"out4/{name}.json"))
    if d["kind"] == "table": return M.table_policy({k: [tuple(x) for x in v] for k, v in d["policy"].items()}, rng)
    return M.net_policy(load_net(d["policy"]), d["kind"] == "greedy", np.random.default_rng(seed))
BOTS = ["cfr", "dqn", "ppo", "nfsp", "ismcts", "mc", "math", "random"]
pol = {b: make(b, i) for i, b in enumerate(BOTS)}
rng = random.Random(7); played = {b: 0 for b in BOTS}; won = {b: 0 for b in BOTS}; t0 = time.time()
for g in range(GAMES):
    seats = rng.sample(BOTS, 4); w = M.play_game([pol[s] for s in seats], rng)
    for s in seats: played[s] += 1
    won[seats[w]] += 1
    if g % 500 == 499: print(g + 1, {b: round(won[b] / played[b], 3) for b in BOTS}, round(time.time() - t0), flush=True)
win = {b: round(won[b] / played[b], 3) for b in BOTS}
json.dump({"win": win, "played": played}, open("out4/tournament4.json", "w"), indent=1)
for b in sorted(BOTS, key=lambda b: -win[b]): print(f"{b:>8} {win[b]:.3f} ({played[b]} games)")
