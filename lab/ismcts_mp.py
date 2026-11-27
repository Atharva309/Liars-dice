import math, random
import mp as M
from mp import bq, bf, CALL, legal
HANDS = {1: [(a,) for a in range(1, 7)], 2: [(a, b) for a in range(1, 7) for b in range(a, 7)]}
HP = {1: [1/6]*6, 2: [(1/36 if a == b else 2/36) for a, b in HANDS[2]]}
def make_ismcts(sims=1000, c=0.5, seed=0):
    rng = random.Random(seed)
    def act(o, acts, ctx):
        hand, mn, ot, no, last, prev = o
        if len(acts) == 1: return acts[0]
        me = ctx["seat"]; seats = ctx["seats"]; counts = ctx["counts"]; hist = ctx["history"]
        i0 = seats.index(me); order = seats[i0:] + seats[:i0]; k = len(order); total = sum(counts)
        cand = {}
        for s in order[1:]:
            bids = [a for (bs, a) in hist if bs == s]; n = counts[s]
            ws = [p * math.prod(1 + 1.5 * h.count(bf(b)) for b in bids) for h, p in zip(HANDS[n], HP[n])]
            cand[s] = (HANDS[n], ws)
        N, W = {}, {}
        for _ in range(sims):
            hands = [hand] + [rng.choices(*cand[s])[0] for s in order[1:]]
            node = (); l = last; mover = 0; bidder = k - 1 if last >= 0 else None; path = []
            while True:
                la = legal(l, total); un = [a for a in la if node + (a,) not in N]
                if un:
                    a = rng.choice(un); node += (a,); N[node] = 0; W[node] = [0.0] * k; path.append(node); break
                t = sum(N[node + (a,)] for a in la)
                a = max(la, key=lambda a: W[node + (a,)][mover] / N[node + (a,)] + c * math.sqrt(math.log(t) / N[node + (a,)]))
                node += (a,); path.append(node)
                if a == CALL: break
                l, bidder, mover = a, mover, (mover + 1) % k
            a = node[-1]
            while a != CALL:
                l, bidder, mover = a, mover, (mover + 1) % k
                a = M.math_bot((hands[mover], len(hands[mover]), total - len(hands[mover]), k - 1, l, -1), legal(l, total))
            cnt = sum(h.count(bf(l)) for h in hands); loser = mover if cnt >= bq(l) else bidder
            r = [M.reward(s, loser, k) for s in range(k)]
            for n in path:
                N[n] += 1; w = W[n]
                for s in range(k): w[s] += r[s]
        return max(acts, key=lambda a: N.get((a,), 0))
    return act
if __name__ == "__main__":
    import time
    for sims, c in [(1000, 0.5), (1000, 1.0)]:
        rng = random.Random(0); w = 0; n = 120; t = time.time()
        for g in range(n):
            ag = [M.math_bot] * 4; ag[g % 4] = make_ismcts(sims, c, g); w += M.play_game(ag, rng) == g % 4
        print(sims, c, "win% vs 3 math:", round(w / n, 3), "sec/game", round((time.time() - t) / n, 2), flush=True)
