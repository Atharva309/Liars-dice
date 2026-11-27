"""Liar's Dice, 2-4 players, 2 dice each, no wilds. Turn order goes round the table.
'Liar!' is a challenge between the caller and the last bidder; the loser drops a die and starts the next round."""
import math, random, json, sys, time, numpy as np
from functools import lru_cache
from nn import MLP, masked_softmax

MAXD = 8; CALL = 48; NA = 49
bq = lambda k: k // 6 + 1; bf = lambda k: k % 6 + 1
def legal(last, total):
    a = list(range(last + 1, total * 6))
    if last >= 0: a.append(CALL)
    return a
@lru_cache(maxsize=None)
def p_true(q, f, hand, others):
    need = q - hand.count(f)
    if need <= 0: return 1.0
    if need > others: return 0.0
    return sum(math.comb(others, k) * (1/6)**k * (5/6)**(others-k) for k in range(need, others + 1))
def roll(n, rng): return tuple(sorted(rng.randint(1, 6) for _ in range(n)))
# obs = (hand, my_n, others_total, n_opp, last, prev)
def tkey(o): return f"{''.join(map(str, o[0]))}|{o[1]}|{o[2]}|{o[4]}"
def encode(o):
    hand, mn, ot, no, last, prev = o
    x = np.zeros(115, np.float32)
    for d in hand: x[d - 1] += 0.5
    x[6 + mn - 1] = 1; x[8 + ot - 1] = 1; x[14 + no - 1] = 1; x[17 + last + 1] = 1; x[66 + prev + 1] = 1
    return x

def play_round(agents, counts, starter, rng, hist_out=None):
    act_seats = [s for s in range(len(counts)) if counts[s] > 0]
    hands = {s: roll(counts[s], rng) for s in act_seats}; total = sum(counts)
    last = prev = -1; seat = starter; bidder = None; hist = []
    while True:
        others = total - counts[seat]
        o = (hands[seat], counts[seat], others, len(act_seats) - 1, last, prev)
        acts = legal(last, total)
        a = agents[seat](o, acts, {"history": hist, "seat": seat, "counts": counts, "seats": act_seats})
        if hist_out is not None: hist_out.append((seat, o, a))
        if a == CALL:
            n = sum(h.count(bf(last)) for h in hands.values())
            return (seat if n >= bq(last) else bidder), hands
        hist.append((seat, a)); prev, last, bidder = last, a, seat
        i = act_seats.index(seat); seat = act_seats[(i + 1) % len(act_seats)]

def play_game(agents, rng, n=4, log=None):
    counts = [2] * n; starter = rng.randrange(n)
    while sum(c > 0 for c in counts) > 1:
        loser, _ = play_round(agents, counts, starter, rng, log)
        counts[loser] -= 1; starter = loser
        while counts[starter] == 0: starter = (starter + 1) % n
    return next(s for s in range(n) if counts[s] > 0)

def math_bot(o, acts, ctx=None):
    hand, mn, ot, no, last, prev = o
    best, bp = None, -1.0
    for b in acts:
        if b == CALL: continue
        p = p_true(bq(b), bf(b), hand, ot)
        if p > bp + 1e-9: best, bp = b, p
    if last >= 0:
        pl = p_true(bq(last), bf(last), hand, ot)
        if best is None or pl < 1 - bp: return CALL
    return best
def make_random(rng):
    def f(o, acts, ctx=None):
        if CALL in acts and (len(acts) == 1 or rng.random() < 1/3): return CALL
        return rng.choice([a for a in acts if a != CALL])
    return f

# ---------- training helpers ----------
def deal(rng):
    k = rng.choice((2, 3, 4, 4)); counts = [rng.choice((1, 2)) for _ in range(k)]
    return counts, [roll(c, rng) for c in counts]
def reward(s, loser, k): return -1.0 if s == loser else 1.0 / (k - 1)
def selfplay_round(choose, rng):
    counts, hands = deal(rng); k = len(counts); total = sum(counts)
    seat = rng.randrange(k); last = prev = -1; bidder = None; traj = [[] for _ in range(k)]
    while True:
        o = (hands[seat], counts[seat], total - counts[seat], k - 1, last, prev); acts = legal(last, total)
        a = choose(seat, o, acts); traj[seat].append((o, a))
        if a == CALL:
            n = sum(h.count(bf(last)) for h in hands)
            return traj, (seat if n >= bq(last) else bidder), k
        prev, last, bidder = last, a, seat; seat = (seat + 1) % k

def eval_vs_math(policy, n=600, seed=0):
    rng = random.Random(seed); w = 0
    for g in range(n):
        me = g % 4; ag = [math_bot] * 4; ag[me] = policy
        w += play_game(ag, rng) == me
    return w / n
def curve_point(curve, step, policy, t0):
    curve.append({"step": step, "vs_math": round(eval_vs_math(policy), 3), "sec": round(time.time() - t0)}); print(curve[-1], flush=True)

def table_policy(tab, rng=random):
    def f(o, acts, ctx=None):
        p = tab.get(tkey(o))
        if p is None: return math_bot(o, acts)
        r = rng.random(); c = 0
        for a, x in p:
            c += x
            if r < c and a in acts: return a
        return max(p, key=lambda t: t[1])[0]
    return f
def net_policy(net, greedy, rng=np.random.default_rng(9)):
    def f(o, acts, ctx=None):
        out = net.forward(encode(o)[None])[0]
        if greedy: return max(acts, key=lambda a: out[a])
        m = np.zeros(NA, bool); m[acts] = True; p = masked_softmax(out[None], m[None])[0]
        return int(rng.choice(NA, p=p))
    return f
def save(name, curve, policy_json, kind):
    json.dump({"kind": kind, "policy": policy_json, "curve": curve}, open(f"out4/{name}.json", "w"), separators=(",", ":"))

# ---------- CFR: multiplayer outcome-sampling MCCFR (the Pluribus family) ----------
def abs_legal(last, total):
    a = legal(last, total); b = [x for x in a if x != CALL][:8]
    return b + ([CALL] if CALL in a else [])

def train_cfr(iters=6_000_000, eps=0.6):
    rng = random.Random(0); R, S = {}, {}; curve = []; t0 = time.time()
    def rm(r):
        p = np.maximum(r, 0); s = p.sum(); return p / s if s > 0 else np.full(len(r), 1 / len(r))
    def tab():
        out = {}
        for k, s in S.items():
            if s.sum() <= 0: continue
            acts = abs_legal(int(k.split("|")[3]), int(k.split("|")[1]) + int(k.split("|")[2])); p = s / s.sum()
            out[k] = [(a, round(float(x), 3)) for a, x in zip(acts, p) if x >= 0.01]
        return out
    def ep(hands, counts, seat, last, prev, bidder, upd, my_r, opp_r, s_r, t):
        k = len(counts); total = sum(counts)
        o = (hands[seat], counts[seat], total - counts[seat], k - 1, last, prev); acts = abs_legal(last, total); key = tkey(o)
        if key not in R: R[key] = np.zeros(len(acts)); S[key] = np.zeros(len(acts))
        pol = rm(R[key]); n = len(acts)
        sp = eps / n + (1 - eps) * pol if seat == upd else pol
        ai = min(int(np.searchsorted(np.cumsum(sp), rng.random())), n - 1); a = acts[ai]
        if a == CALL:
            cnt = sum(h.count(bf(last)) for h in hands); loser = seat if cnt >= bq(last) else bidder
            cv = reward(upd, loser, k)
        else:
            cv = ep(hands, counts, (seat + 1) % k, a, last, seat, upd,
                    my_r * (pol[ai] if seat == upd else 1), opp_r * (pol[ai] if seat != upd else 1), s_r * sp[ai], t)
        child = np.zeros(n); child[ai] = cv / sp[ai]; v = pol[ai] * child[ai]
        if seat == upd:
            R[key] += (child - v) * opp_r / s_r; S[key] += my_r * pol / s_r
        return v
    for it in range(1, iters + 1):
        counts, hands = deal(rng); k = len(counts)
        ep(hands, counts, rng.randrange(k), -1, -1, None, rng.randrange(k), 1.0, 1.0, 1.0, it)
        if it % 500_000 == 0 or it == 100_000: curve_point(curve, it, table_policy(tab()), t0)
    save("cfr", curve, tab(), "table")

# ---------- Monte Carlo control ----------
def train_mc(rounds=3_000_000, alpha=0.02):
    rng = random.Random(1); Q = {}; curve = []; t0 = time.time()
    def tab(): return {k: [(max(q, key=q.get), 1.0)] for k, q in Q.items()}
    for r in range(1, rounds + 1):
        e = max(0.05, 1 - r / (rounds * 0.6))
        def choose(s, o, acts):
            q = Q.setdefault(tkey(o), {a: 0.0 for a in acts})
            return rng.choice(acts) if rng.random() < e else max(acts, key=q.get)
        traj, loser, k = selfplay_round(choose, rng)
        for s in range(k):
            G = reward(s, loser, k)
            for o, a in traj[s]: q = Q[tkey(o)]; q[a] += alpha * (G - q[a])
        if r % 500_000 == 0 or r == 50_000: curve_point(curve, r, table_policy(tab()), t0)
    save("mc", curve, tab(), "table")

# ---------- DQN pieces ----------
class Replay:
    def __init__(s, cap=150_000, d=115):
        s.cap, s.i, s.n = cap, 0, 0
        s.s = np.zeros((cap, d), np.float32); s.a = np.zeros(cap, np.int64); s.r = np.zeros(cap, np.float32)
        s.s2 = np.zeros((cap, d), np.float32); s.m2 = np.zeros((cap, NA), bool); s.d = np.zeros(cap, np.float32)
    def add(s, *v):
        i = s.i; s.s[i], s.a[i], s.r[i], s.s2[i], s.m2[i], s.d[i] = v; s.i = (i + 1) % s.cap; s.n = min(s.n + 1, s.cap)
    def sample(s, b, r): idx = r.integers(0, s.n, b); return s.s[idx], s.a[idx], s.r[idx], s.s2[idx], s.m2[idx], s.d[idx]
def mask(acts): m = np.zeros(NA, bool); m[acts] = True; return m
def push(buf, traj, loser, k):
    for s in range(k):
        G = reward(s, loser, k); tr = traj[s]
        for t, (o, a) in enumerate(tr):
            if t + 1 < len(tr): o2 = tr[t+1][0]; buf.add(encode(o), a, 0.0, encode(o2), mask(legal(o2[4], o2[1] + o2[2])), 0.0)
            else: buf.add(encode(o), a, G, np.zeros(115, np.float32), np.ones(NA, bool), 1.0)
def dqn_update(q, qt, buf, nr, b=128):
    s, a, r, s2, m2, d = buf.sample(b, nr); a2 = np.where(m2, q.forward(s2), -1e9).argmax(1)
    tgt = r + (1 - d) * qt.forward(s2)[np.arange(b), a2]
    out = q.forward(s, cache=True); g = np.zeros_like(out); g[np.arange(b), a] = np.clip(out[np.arange(b), a] - tgt, -1, 1) / b; q.backward(g)

def train_dqn(rounds=150_000):
    rng = random.Random(2); nr = np.random.default_rng(2); curve = []; t0 = time.time()
    q = MLP([115, 128, 128, NA], 2, 5e-4); qt = MLP([115, 128, 128, NA], 2); qt.copy_from(q); buf = Replay(); ups = 0
    for r in range(1, rounds + 1):
        e = max(0.05, 1 - r / (rounds * 0.5))
        def choose(s, o, acts):
            if rng.random() < e: return rng.choice(acts)
            v = q.forward(encode(o)[None])[0]; return max(acts, key=lambda a: v[a])
        traj, loser, k = selfplay_round(choose, rng); push(buf, traj, loser, k)
        if buf.n > 3000:
            dqn_update(q, qt, buf, nr); ups += 1
            if ups % 500 == 0: qt.copy_from(q)
        if r % 15000 == 0 or r == 3000: curve_point(curve, r, net_policy(q, True), t0)
    save("dqn", curve, q.export(), "greedy")

def train_ppo(iters=300, per=1024):
    rng = random.Random(3); nr = np.random.default_rng(3); curve = []; t0 = time.time()
    pi = MLP([115, 128, 128, NA], 3, 3e-4); vf = MLP([115, 128, 128, 1], 4, 1e-3)
    for it in range(1, iters + 1):
        S, Mk, A, LP, G = [], [], [], [], []
        for _ in range(per):
            buf = []
            def choose(s, o, acts):
                x = encode(o); m = mask(acts); p = masked_softmax(pi.forward(x[None]), m[None])[0]
                a = int(nr.choice(NA, p=p)); buf.append((s, x, m, a, np.log(p[a] + 1e-12))); return a
            traj, loser, k = selfplay_round(choose, rng)
            for s, x, m, a, lp in buf: S.append(x); Mk.append(m); A.append(a); LP.append(lp); G.append(reward(s, loser, k))
        S, Mk, A, LP, G = map(np.array, (S, Mk, A, LP, G)); N = len(S); ec = 0.02 * (1 - it / iters) + 0.002
        for _ in range(4):
            for idx in np.array_split(nr.permutation(N), max(1, N // 512)):
                s, m, a, lp, g = S[idx], Mk[idx], A[idx], LP[idx], G[idx]; b = len(idx)
                v = vf.forward(s, cache=True)[:, 0]; vf.backward(((v - g) / b)[:, None])
                adv = g - v; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                p = masked_softmax(pi.forward(s, cache=True), m); ratio = np.exp(np.log(p[np.arange(b), a] + 1e-12) - lp)
                use = ~(((ratio > 1.2) & (adv > 0)) | ((ratio < 0.8) & (adv < 0)))
                oh = np.zeros_like(p); oh[np.arange(b), a] = 1
                logp = np.log(p + 1e-12) * m; H = -(p * logp).sum(1, keepdims=True)
                pi.backward(((-(ratio * adv * use))[:, None] * (oh - p) + ec * p * (logp + H)) * m / b)
        if it % 25 == 0 or it == 2: curve_point(curve, it * per, net_policy(pi, False), t0)
    save("ppo", curve, pi.export(), "softmax")

def train_nfsp(rounds=250_000, eta=0.25):
    rng = random.Random(5); nr = np.random.default_rng(5); curve = []; t0 = time.time()
    q = MLP([115, 128, 128, NA], 5, 5e-4); qt = MLP([115, 128, 128, NA], 5); qt.copy_from(q)
    avg = MLP([115, 128, 128, NA], 6, 5e-4); buf = Replay()
    cap = 200_000; SLs = np.zeros((cap, 115), np.float32); SLm = np.zeros((cap, NA), bool); SLa = np.zeros(cap, np.int64); seen = 0; ups = 0
    for r in range(1, rounds + 1):
        mode = [rng.random() < eta for _ in range(4)]
        def choose(s, o, acts):
            nonlocal seen
            x = encode(o); m = mask(acts)
            if mode[s]:
                if rng.random() < 0.06: return rng.choice(acts)
                v = q.forward(x[None])[0]; a = max(acts, key=lambda k: v[k])
                j = seen if seen < cap else nr.integers(0, seen + 1)
                if j < cap: SLs[j], SLm[j], SLa[j] = x, m, a
                seen += 1; return a
            p = masked_softmax(avg.forward(x[None]), m[None])[0]; return int(nr.choice(NA, p=p))
        traj, loser, k = selfplay_round(choose, rng); push(buf, traj, loser, k)
        if buf.n > 3000:
            dqn_update(q, qt, buf, nr); ups += 1
            if ups % 500 == 0: qt.copy_from(q)
        if seen > 1000:
            idx = nr.integers(0, min(seen, cap), 128); p = masked_softmax(avg.forward(SLs[idx], cache=True), SLm[idx])
            oh = np.zeros_like(p); oh[np.arange(128), SLa[idx]] = 1; avg.backward((p - oh) * SLm[idx] / 128)
        if r % 25000 == 0 or r == 3000: curve_point(curve, r, net_policy(avg, False), t0)
    save("nfsp", curve, avg.export(), "softmax")

if __name__ == "__main__":
    kw = {k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])}; globals()["train_" + sys.argv[1]](**kw)
