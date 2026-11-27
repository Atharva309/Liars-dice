"""Liar's Dice, 2-4 players, up to 5 dice each (20 on the table).
The 2-dice agents memorised the raw game; that doesn't scale. Here every agent sees the game through
a shortlist of sensible moves and a few odds-based numbers per move, so one trained agent plays any dice count.
Learning rules are unchanged: regret (CFR), win/loss averages (MC), Q-learning (DQN), policy gradient (PPO), NFSP."""
import math, random, json, sys, time, base64, os, numpy as np
from nn import MLP, masked_softmax
MAXD, CALL, NF, MAXC, NB = 5, 999, 15, 19, 31
PT = [[1.0 if nd <= 0 else (0.0 if nd > o else sum(math.comb(o, k) * (1/6)**k * (5/6)**(o-k) for k in range(nd, o + 1))) for nd in range(22)] for o in range(16)]
def tail(o, nd): return 1.0 if nd <= 0 else (0.0 if nd > o else PT[o][nd])
N50 = [max(nd for nd in range(o + 1) if PT[o][nd] >= .5) for o in range(16)]
N25 = [max(nd for nd in range(o + 1) if PT[o][nd] >= .25) for o in range(16)]
def counts(hand):
    c = [0] * 6
    for d in hand: c[d - 1] += 1
    return c
def cands(c, n, others, last):
    """Shortlist: per face, the cheapest legal raise, the biggest claim that's >=50% likely, and the biggest that's >=25%."""
    total = n + others; out = []; lq, lf = (last // 6 + 1, last % 6 + 1) if last >= 0 else (0, 0)
    for f in range(1, 7):
        q0 = lq if (last >= 0 and f > lf) else lq + 1
        if q0 > total: continue
        qc = c[f-1] + N50[others]; qa = c[f-1] + N25[others]
        out.append((q0 - 1) * 6 + f - 1)
        if q0 < qc <= total: out.append((qc - 1) * 6 + f - 1)
        if qa > qc and q0 < qa <= total: out.append((qa - 1) * 6 + f - 1)
    out.sort()
    if last >= 0: out.append(CALL)
    return out
def feats(c, n, others, nopp, last, prev, acts):
    total = n + others; sd = math.sqrt(others * 5 / 36) or 1.0; mu = others / 6
    lq, lf = (last // 6 + 1, last % 6 + 1) if last >= 0 else (0, 0)
    if last >= 0: nl = lq - c[lf-1]; st = [n / 5, others / 15, nopp / 3, 1.0, tail(others, nl), max(-3.0, min(3.0, (nl - mu) / sd)) / 3, (max(0, min(3, lq - (prev // 6 + 1))) / 3 if prev >= 0 else 0.0)]
    else: st = [n / 5, others / 15, nopp / 3, 0.0, 0.0, 0.0, 0.0]
    rows = []
    for a in acts:
        if a == CALL: rows.append(st + [1.0, 0, 0, 0, 0, 0, 0, 0]); continue
        q, f = a // 6 + 1, a % 6 + 1; nd = q - c[f-1]
        rows.append(st + [0.0, tail(others, nd), max(-3.0, min(3.0, (nd - mu) / sd)) / 3, max(0, min(4, q - lq)) / 4, 1.0 if (last >= 0 and f == lf) else 0.0, c[f-1] / n, f / 6, q / total])
    return np.array(rows, np.float32)
PLB, PAB = (.05, .2, .4, .6, .8), (.15, .35, .55, .8)
def skey(c, n, others, nopp, last):
    b = 0 if last < 0 else 1 + sum(tail(others, last // 6 + 1 - c[last % 6]) >= t for t in PLB)
    return f"{b}|{0 if others <= 2 else 1 if others <= 5 else 2 if others <= 9 else 3}|{n}|{nopp}"
def bmap(c, others, last, acts):
    """Group shortlist moves into 31 kinds (call, or how likely x how big a jump x same face). One move per kind."""
    lq, lf = (last // 6 + 1, last % 6 + 1) if last >= 0 else (0, 0); out = {}; best = {}
    for a in acts:
        if a == CALL: out[0] = a; continue
        q, f = a // 6 + 1, a % 6 + 1; pa = tail(others, q - c[f-1]); dq = q - lq
        b = 1 + sum(pa >= t for t in PAB) * 6 + (0 if dq <= 1 else 1 if dq == 2 else 2) * 2 + (1 if (last >= 0 and f == lf) else 0)
        if b not in out or pa > best[b] + 1e-9: out[b] = a; best[b] = pa
    return out
def roll(n, rng): return tuple(sorted(rng.randint(1, 6) for _ in range(n)))
def reward(s, loser, k): return -1.0 if s == loser else 1.0 / (k - 1)
def deal(rng):
    k = rng.choice((2, 3, 4, 4)); cnt = [rng.randint(1, MAXD) for _ in range(k)]
    return cnt, [roll(x, rng) for x in cnt]
def selfplay_round(choose, rng, mix=0.0):
    cnt, hands = deal(rng); k = len(cnt); total = sum(cnt); C = [counts(h) for h in hands]
    bots = [rng.random() < mix for _ in range(k)] if mix else None
    if bots and all(bots): bots[rng.randrange(k)] = False
    seat = rng.randrange(k); last = prev = -1; bidder = None
    while True:
        if bots and bots[seat]: a = math_bot((hands[seat], cnt[seat], total - cnt[seat], k - 1, last, prev))
        else: a = choose(seat, C[seat], cnt[seat], total - cnt[seat], k - 1, last, prev)
        if a == CALL: return (seat if sum(c[last % 6] for c in C) >= last // 6 + 1 else bidder), k
        prev, last, bidder = last, a, seat; seat = (seat + 1) % k
def play_game(agents, rng, n=4, start=5):
    cnt = [start] * n; starter = rng.randrange(n)
    while sum(x > 0 for x in cnt) > 1:
        act = [s for s in range(n) if cnt[s] > 0]; hands = {s: roll(cnt[s], rng) for s in act}; total = sum(cnt)
        last = prev = -1; seat = starter; bidder = None
        while True:
            a = agents[seat]((hands[seat], cnt[seat], total - cnt[seat], len(act) - 1, last, prev))
            if a == CALL:
                assert last >= 0
                loser = seat if sum(h.count(last % 6 + 1) for h in hands.values()) >= last // 6 + 1 else bidder; break
            assert last < a < total * 6, (a, last, total)
            prev, last, bidder = last, a, seat; seat = act[(act.index(seat) + 1) % len(act)]
        cnt[loser] -= 1; starter = loser
        while cnt[starter] == 0: starter = (starter + 1) % n
    return next(s for s in range(n) if cnt[s] > 0)
def math_bot(o, ctx=None):
    hand, n, others, nopp, last, prev = o; c = counts(hand); best, bp = None, -1.0
    for b in range(last + 1, (n + others) * 6):
        p = tail(others, b // 6 + 1 - c[b % 6])
        if p > bp + 1e-9: best, bp = b, p
    if last >= 0 and (best is None or tail(others, last // 6 + 1 - c[last % 6]) < 1 - bp): return CALL
    return best
def eval_vs_math(policy, n=240, seed=0, start=5):
    rng = random.Random(seed); w = 0
    for g in range(n):
        ag = [math_bot] * 4; ag[g % 4] = policy; w += play_game(ag, rng, 4, start) == g % 4
    return w / n
def point(curve, step, policy, t0):
    curve.append({"step": step, "vs_math": round(eval_vs_math(policy), 3), "sec": round(time.time() - t0)}); print(curve[-1], flush=True)
    return curve[-1]["vs_math"] >= max(c["vs_math"] for c in curve)
def snap(net): c = MLP([1]); c.copy_from(net); return c
def net_policy(net, greedy, nr=np.random.default_rng(9)):
    def f(o, ctx=None):
        hand, n, others, nopp, last, prev = o; c = counts(hand); acts = cands(c, n, others, last)
        if len(acts) == 1: return acts[0]
        z = net.forward(feats(c, n, others, nopp, last, prev, acts))[:, 0]
        if greedy: return acts[int(z.argmax())]
        p = np.exp(z - z.max()); p /= p.sum(); return acts[int(nr.choice(len(acts), p=p))]
    return f
def table_policy(T, greedy, rng=random.Random(4)):
    def f(o, ctx=None):
        hand, n, others, nopp, last, prev = o; c = counts(hand); acts = cands(c, n, others, last); bm = bmap(c, others, last, acts); row = T.get(skey(c, n, others, nopp, last))
        if row is None: return math_bot(o)
        bs = sorted(bm)
        if greedy: return bm[max(bs, key=lambda b: row[b])]
        w = [max(row[b], 0) for b in bs]; s = sum(w)
        return bm[rng.choices(bs, w)[0]] if s > 0 else math_bot(o)
    return f
def export_net(net):
    L = []
    for W, b in zip(net.W, net.b):
        s = float(max(np.abs(W).max(), np.abs(b).max())) / 32767; q = np.round(np.concatenate([W.ravel(), b]) / s).astype('<i2')
        L.append({"i": W.shape[0], "o": W.shape[1], "s": s, "d": base64.b64encode(q.tobytes()).decode()})
    return L
def save(name, obj, curve):
    obj["curve"] = curve; json.dump(obj, open(f"out5/{name}.json", "w"), separators=(",", ":"))

# ---------- Monte Carlo control (table over move kinds) ----------
def train_mc(rounds=900_000, alpha=0.03):
    rng = random.Random(1); Q = {}; curve = []; t0 = time.time()
    for r in range(1, rounds + 1):
        eps = max(.05, 1 - r / (rounds * .6)); steps = [[], [], [], []]
        def choose(s, c, n, o, no, last, prev):
            bm = bmap(c, o, last, cands(c, n, o, last)); k = skey(c, n, o, no, last); q = Q.get(k)
            if q is None: q = Q[k] = np.zeros(NB)
            bs = list(bm); b = rng.choice(bs) if rng.random() < eps else max(bs, key=lambda x: q[x]); steps[s].append((q, b)); return bm[b]
        loser, k = selfplay_round(choose, rng, .25)
        for s in range(k):
            G = reward(s, loser, k)
            for q, b in steps[s]: q[b] += alpha * (G - q[b])
        if r % (rounds // 8) == 0 and point(curve, r, table_policy(Q, True), t0): best = {k: v.copy() for k, v in Q.items()}
    save("mc", {"kind": "mcq", "t": {k: [round(float(x), 3) for x in v] for k, v in best.items()}}, curve)

# ---------- CFR: external-sampling MCCFR over the same abstraction (the method Pluribus used) ----------
def train_cfr(seconds=420):
    rng = random.Random(0); R, S = {}, {}; curve = []; t0 = time.time(); best = None; it = 0; nxt = seconds / 8
    def walk(C, cnt, total, seat, last, prev, bidder, upd, depth, t):
        k = len(cnt); n = cnt[seat]; others = total - n; c = C[seat]
        bm = bmap(c, others, last, cands(c, n, others, last)); bs = sorted(bm); key = skey(c, n, others, k - 1, last)
        r = R.get(key)
        if r is None: r = R[key] = np.zeros(NB); S[key] = np.zeros(NB)
        pos = np.maximum(r[bs], 0); tt = pos.sum(); m = len(bs); pol = pos / tt if tt > 0 else np.full(m, 1 / m)
        def val(a):
            if a == CALL: return reward(upd, seat if sum(x[last % 6] for x in C) >= last // 6 + 1 else bidder, k)
            return walk(C, cnt, total, (seat + 1) % k, a, last, seat, upd, depth + (seat == upd), t)
        if seat == upd and depth < 2:          # try every move here and learn which ones I regret not taking
            vals = np.array([val(bm[b]) for b in bs]); v = float(pol @ vals); r[bs] += vals - v; return v
        if seat != upd: S[key][bs] += t * pol   # everyone else just plays their current strategy
        return val(bm[bs[min(int(np.searchsorted(np.cumsum(pol), rng.random())), m - 1)]])
    while time.time() - t0 < seconds:
        it += 1; cnt, hands = deal(rng); k = len(cnt); C = [counts(h) for h in hands]
        for upd in range(k): walk(C, cnt, sum(cnt), rng.randrange(k), -1, -1, None, upd, 0, it)
        if time.time() - t0 >= nxt:
            nxt += seconds / 8
            cur = {k: np.maximum(v, 0) for k, v in R.items()}   # final-iterate strategy: at 4 players the long-run average plays far worse
            if point(curve, it, table_policy(cur, False), t0): best = cur
    save("cfr", {"kind": "cfrs", "t": {k: [round(float(x), 3) for x in (v / v.sum() if v.sum() > 0 else v)] for k, v in best.items()}}, curve)

# ---------- shared pieces for the neural agents: one small net scores each shortlisted move ----------
class Replay:
    def __init__(s, cap):
        s.cap, s.i, s.n = cap, 0, 0; s.f = np.zeros((cap, NF), np.float32); s.r = np.zeros(cap, np.float32)
        s.x2 = np.zeros((cap, MAXC, NF), np.float32); s.m2 = np.zeros((cap, MAXC), bool); s.d = np.zeros(cap, np.float32)
    def add(s, f, r, x2, d):
        i = s.i; s.f[i] = f; s.r[i] = r; s.d[i] = d; s.m2[i] = False
        if x2 is not None: s.x2[i, :len(x2)] = x2; s.m2[i, :len(x2)] = True
        s.i = (i + 1) % s.cap; s.n = min(s.n + 1, s.cap)
    def sample(s, b, nr): idx = nr.integers(0, s.n, b); return s.f[idx], s.r[idx], s.x2[idx], s.m2[idx], s.d[idx]
def push(buf, steps, loser, k):
    for s in range(k):
        G = reward(s, loser, k); tr = steps[s]
        for t, (F, ai) in enumerate(tr):
            if t + 1 < len(tr): buf.add(F[ai], 0.0, tr[t + 1][0], 0.0)
            else: buf.add(F[ai], G, None, 1.0)
def dqn_update(q, qt, buf, nr, b=128):
    f, r, x2, m2, d = buf.sample(b, nr); flat = x2.reshape(-1, NF)
    a2 = np.where(m2, q.forward(flat).reshape(b, MAXC), -1e9).argmax(1)
    tgt = r + (1 - d) * qt.forward(flat).reshape(b, MAXC)[np.arange(b), a2]
    out = q.forward(f, cache=True); q.backward((np.clip(out[:, 0] - tgt, -1, 1) / b)[:, None])
def train_dqn(rounds=110_000):
    rng = random.Random(2); nr = np.random.default_rng(2); curve = []; t0 = time.time()
    q = MLP([NF, 48, 48, 1], 2, 1e-3); qt = MLP([NF, 48, 48, 1], 2); qt.copy_from(q); buf = Replay(60_000); ups = 0
    for r in range(1, rounds + 1):
        eps = max(.05, 1 - r / (rounds * .5)); steps = [[], [], [], []]
        def choose(s, c, n, o, no, last, prev):
            acts = cands(c, n, o, last); F = feats(c, n, o, no, last, prev, acts)
            ai = rng.randrange(len(acts)) if rng.random() < eps else int(q.forward(F)[:, 0].argmax()); steps[s].append((F, ai)); return acts[ai]
        loser, k = selfplay_round(choose, rng, .25); push(buf, steps, loser, k)
        if buf.n > 3000:
            dqn_update(q, qt, buf, nr); ups += 1
            if ups % 400 == 0: qt.copy_from(q)
        if r % (rounds // 10) == 0 and point(curve, r, net_policy(q, True), t0): best = snap(q)
    save("dqn", {"kind": "q", "layers": export_net(best)}, curve)
def train_ppo(iters=150, per=768):
    rng = random.Random(3); nr = np.random.default_rng(3); curve = []; t0 = time.time()
    pi = MLP([NF, 48, 48, 1], 3, 1e-3); vf = MLP([7, 32, 32, 1], 4, 2e-3)
    for it in range(1, iters + 1):
        X, Mk, A, LP, G = [], [], [], [], []
        for _ in range(per):
            buf = []
            def choose(s, c, n, o, no, last, prev):
                acts = cands(c, n, o, last); F = feats(c, n, o, no, last, prev, acts); z = pi.forward(F)[:, 0]; p = np.exp(z - z.max()); p /= p.sum()
                ai = int(nr.choice(len(acts), p=p)); buf.append((s, F, ai, math.log(p[ai] + 1e-12))); return acts[ai]
            loser, k = selfplay_round(choose, rng)
            for s, F, ai, lp in buf:
                x = np.zeros((MAXC, NF), np.float32); x[:len(F)] = F; m = np.zeros(MAXC, bool); m[:len(F)] = True
                X.append(x); Mk.append(m); A.append(ai); LP.append(lp); G.append(reward(s, loser, k))
        X, Mk, A, LP, G = np.array(X), np.array(Mk), np.array(A), np.array(LP), np.array(G, np.float32); N = len(X); ec = .02 * (1 - it / iters) + .002
        for _ in range(4):
            for idx in np.array_split(nr.permutation(N), max(1, N // 512)):
                x, m, a, lp, g = X[idx], Mk[idx], A[idx], LP[idx], G[idx]; b = len(idx)
                v = vf.forward(x[:, 0, :7], cache=True)[:, 0]; vf.backward(((v - g) / b)[:, None])
                adv = g - v; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                p = masked_softmax(pi.forward(x.reshape(-1, NF), cache=True).reshape(b, MAXC), m); ratio = np.exp(np.log(p[np.arange(b), a] + 1e-12) - lp)
                use = ~(((ratio > 1.2) & (adv > 0)) | ((ratio < .8) & (adv < 0))); oh = np.zeros_like(p); oh[np.arange(b), a] = 1
                logp = np.log(p + 1e-12) * m; H = -(p * logp).sum(1, keepdims=True)
                pi.backward((((-(ratio * adv * use))[:, None] * (oh - p) + ec * p * (logp + H)) * m / b).reshape(-1, 1))
        if it % max(1, iters // 10) == 0 and point(curve, it * per, net_policy(pi, False), t0): best = snap(pi)
    save("ppo", {"kind": "pi", "layers": export_net(best)}, curve)
def train_nfsp(rounds=90_000, eta=.25):
    rng = random.Random(5); nr = np.random.default_rng(5); curve = []; t0 = time.time()
    q = MLP([NF, 48, 48, 1], 5, 1e-3); qt = MLP([NF, 48, 48, 1], 5); qt.copy_from(q); avg = MLP([NF, 48, 48, 1], 6, 1e-3); buf = Replay(60_000)
    cap = 80_000; SX = np.zeros((cap, MAXC, NF), np.float32); SM = np.zeros((cap, MAXC), bool); SA = np.zeros(cap, np.int64); seen = 0; ups = 0
    for r in range(1, rounds + 1):
        mode = [rng.random() < eta for _ in range(4)]; steps = [[], [], [], []]
        def choose(s, c, n, o, no, last, prev):
            nonlocal seen
            acts = cands(c, n, o, last); F = feats(c, n, o, no, last, prev, acts)
            if mode[s]:
                if rng.random() < .06: ai = rng.randrange(len(acts))
                else:
                    ai = int(q.forward(F)[:, 0].argmax()); j = seen if seen < cap else int(nr.integers(0, seen + 1))
                    if j < cap: SX[j] = 0; SX[j, :len(F)] = F; SM[j] = False; SM[j, :len(F)] = True; SA[j] = ai
                    seen += 1
            else:
                z = avg.forward(F)[:, 0]; p = np.exp(z - z.max()); p /= p.sum(); ai = int(nr.choice(len(acts), p=p))
            steps[s].append((F, ai)); return acts[ai]
        loser, k = selfplay_round(choose, rng, .25); push(buf, steps, loser, k)
        if buf.n > 3000:
            dqn_update(q, qt, buf, nr); ups += 1
            if ups % 400 == 0: qt.copy_from(q)
        if seen > 1000:
            idx = nr.integers(0, min(seen, cap), 128); p = masked_softmax(avg.forward(SX[idx].reshape(-1, NF), cache=True).reshape(128, MAXC), SM[idx])
            oh = np.zeros_like(p); oh[np.arange(128), SA[idx]] = 1; avg.backward(((p - oh) * SM[idx] / 128).reshape(-1, 1))
        if r % (rounds // 10) == 0 and point(curve, r, net_policy(avg, False), t0): best = snap(avg)
    save("nfsp", {"kind": "pi", "layers": export_net(best)}, curve)
if __name__ == "__main__":
    kw = {k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])}; globals()["train_" + sys.argv[1]](**kw)
