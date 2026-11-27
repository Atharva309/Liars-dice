"""Information-Set MCTS: no training. Each turn: guess opponent dice (weighted by their bids),
search a game tree with UCT, finish simulations with the math bot."""
import math, random
from functools import lru_cache
import engine as E

@lru_cache(maxsize=None)
def pt(q, f, hand, on): return E.p_true(q, f, hand, on)

def fast_math(hand, on, last, total):
    best, bp = None, -1.0
    for b in range(last + 1, total * 6):
        p = pt(b // 6 + 1, b % 6 + 1, hand, on)
        if p > bp + 1e-9: best, bp = b, p
    if last >= 0:
        pl = pt(last // 6 + 1, last % 6 + 1, hand, on)
        if best is None or pl < 1 - bp: return E.CALL
    return best

def make_ismcts(sims=400, c=1.0, seed=0):
    rng = random.Random(seed)
    def act(obs, acts, ctx):
        hand, mn, on, last, prev = obs
        if len(acts) == 1: return acts[0]
        hist = ctx["history"]; opp_bids = hist[-1::-2]
        cands = [(h, p * math.prod(1 + 1.5 * h.count(b % 6 + 1) for b in opp_bids)) for h, p in E.HANDS[on]]
        hs, ws = zip(*cands); total = mn + on
        N, W = {}, {}   # key: action tuple -> visits / value for ROOT player
        for _ in range(sims):
            opp = rng.choices(hs, ws)[0]; hands = (hand, opp)   # 0 = me, 1 = opp
            path, node, l, mover = [], (), last, 0
            # selection / expansion
            while True:
                la = E.legal(l, total)
                untried = [a for a in la if node + (a,) not in N]
                if untried:
                    a = rng.choice(untried); node = node + (a,); path.append(node); N[node] = 0; W[node] = 0.0
                    break
                tot = sum(N[node + (a,)] for a in la); sgn = 1 if mover == 0 else -1
                a = max(la, key=lambda a: sgn * W[node + (a,)] / N[node + (a,)] + c * math.sqrt(math.log(tot) / N[node + (a,)]))
                node = node + (a,); path.append(node)
                if a == E.CALL: break
                l, mover = a, 1 - mover
            # rollout
            a = node[-1]
            while a != E.CALL:
                l, mover = a, 1 - mover
                a = fast_math(hands[mover], (on if mover == 0 else mn), l, total)
            q, f = l // 6 + 1, l % 6 + 1
            true = hand.count(f) + opp.count(f) >= q
            loser = mover if true else 1 - mover      # mover is the caller here
            r = 1.0 if loser == 1 else -1.0
            for n in path: N[n] += 1; W[n] += r
        return max(acts, key=lambda a: N.get((a,), 0))
    return act
