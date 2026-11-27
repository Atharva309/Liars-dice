import json, random, itertools, time, math, sys
import engine as E
from agents import table_agent
from ismcts import make_ismcts, pt

NAMES = {"cfr": "Game Theorist", "dqn": "Exploiter", "ppo": "Gut Player", "nfsp": "Hybrid", "ismcts": "Thinker",
         "mc": "Old-Schooler", "math": "Math Bot", "random": "Random Bot"}
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
GAMES_SEARCH = int(sys.argv[2]) if len(sys.argv) > 2 else 400

tables = {n: json.load(open(f"out/{n}.json"))["policy"] for n in ("cfr", "dqn", "ppo", "nfsp", "mc")}
def make(n, seed):
    rng = random.Random(seed)
    if n in tables: return table_agent(tables[n], rng)
    if n == "ismcts": return make_ismcts(2000, 0.5, seed)
    if n == "math": return E.math_bot
    return lambda o, a, c: E.random_bot(o, a, c, rng)

stats = {n: {"bids": 0, "bluffs": 0, "calls": 0, "calls_right": 0, "call_chances": 0, "bluffs_caught": 0, "bluffs_called": 0} for n in NAMES}
def record(log, names):
    i = 0
    while i < len(log):
        # split into rounds at 'end'
        j = i
        while log[j][0] != "end": j += 1
        rnd, (_, hands, loser) = log[i:j], log[j]
        for k, (seat, obs, a) in enumerate(rnd):
            s = stats[names[seat]]; hand, mn, on, last, prev = obs
            if last >= 0: s["call_chances"] += 1
            if a == E.CALL:
                s["calls"] += 1; s["calls_right"] += loser != seat
            else:
                s["bids"] += 1
                bluff = pt(E.bid_q(a), E.bid_f(a), hand, on) < 0.5
                s["bluffs"] += bluff
                if bluff and k + 1 < len(rnd) and rnd[k+1][2] == E.CALL:
                    s["bluffs_called"] += 1; s["bluffs_caught"] += loser == seat
        i = j + 1

order = list(NAMES); res = {}
t0 = time.time()
for a, b in itertools.combinations(order, 2):
    n = GAMES_SEARCH if "ismcts" in (a, b) else GAMES
    A, B = make(a, 1), make(b, 2); rng = random.Random(hash((a, b)) % 10**6); wa = 0
    for g in range(n):
        log = []; starter = g // 2 % 2
        if g % 2 == 0: wa += E.play_game(A, B, starter, rng, log) == 0; record(log, (a, b))
        else: wa += E.play_game(B, A, starter, rng, log) == 1; record(log, (b, a))
    res[f"{a}|{b}"] = [wa, n]
    print(f"{NAMES[a]:>14} vs {NAMES[b]:<14} {wa/n:.3f}  ({n} games, {time.time()-t0:.0f}s)", flush=True)

# Elo via simple Bradley-Terry fit
elo = {n: 0.0 for n in order}
for _ in range(2000):
    for n in order:
        g = 0
        for k, (w, m) in res.items():
            x, y = k.split("|")
            if n not in (x, y): continue
            o = y if n == x else x; s = w / m if n == x else 1 - w / m
            g += m * (s - 1 / (1 + 10 ** ((elo[o] - elo[n]) / 400)))
        elo[n] += 0.002 * g
mean = sum(elo.values()) / len(elo)
elo = {n: round(1500 + v - mean) for n, v in elo.items()}
json.dump({"names": NAMES, "results": res, "elo": elo, "stats": stats}, open("out/tournament.json", "w"), indent=1)
for n in sorted(order, key=lambda n: -elo[n]): print(f"{NAMES[n]:>14} {elo[n]}")
