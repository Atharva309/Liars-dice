"""Generation 2 of the Liar's Dice bots (files live in out6/). Same six learning methods as before, with five upgrades:
 1. They read the bidding: a fitted table says how many of a face a player probably holds given what they've bid,
    and every probability the bots use is computed from that instead of from pure chance.
 2. They play for the game, not the round: the reward is the change in a player's estimated chance of winning the game.
 3. They train against a crowd: copies of themselves, the previous generation, the Math Bot (and in phase 2, each other).
 4. Bigger networks and a wider shortlist of claims (up to 24 instead of 18), with more numbers describing each.
 5. Best snapshot kept, judged against a table of the previous generation's three best bots."""
import math, random, json, sys, time, base64, os, numpy as np
import gen as G2
from nn import MLP, masked_softmax
CALL, tail, counts = G2.CALL, G2.tail, G2.counts
NF, MAXC, NB, SCALE = 24, 25, 43, 8.0
os.makedirs('out6', exist_ok=True)
BIN = {n: np.array([math.comb(n, c) * (1/6)**c * (5/6)**(n-c) for c in range(n + 1)]) for n in range(1, 6)}
ALPHA = 1.4; POST = {n: {b: {a: BIN[n] for a in (0, 1)} for b in (0, 1, 2)} for n in range(1, 6)}; _TC = {}
def load_model(path='out6/model.json'):
    """alpha: game-win chance ~ dice^alpha share. post[n][b][a]: distribution of how many of a face a player with n dice holds,
    given b = times they bid that face this round (0,1,2+) and a = whether they bid any other face."""
    global ALPHA, POST
    if not os.path.exists(path): return False
    m = json.load(open(path)); ALPHA = m['alpha']; _TC.clear()
    POST = {int(n): {int(b): {int(a): np.array(v) for a, v in d2.items()} for b, d2 in d1.items()} for n, d1 in m['post'].items()}
    return True
load_model()

class X:
    __slots__ = ('hand', 'c', 'n', 'others', 'nopp', 'last', 'prev', 'opp', 'nbids')   # opp: [(dice, face-bid counts)] in turn order after me

def hidden(key):
    """key: sorted tuple of (dice, b, a) per opponent. Returns (tails, need75, need50, need25): tails[m] = P(at least m of the face are hidden)."""
    r = _TC.get(key)
    if r is None:
        d = np.ones(1)
        for nj, b, a in key: d = np.convolve(d, POST[nj][b][a])
        t = np.append(np.cumsum(d[::-1])[::-1], 0.0).tolist(); o = len(t) - 2
        r = _TC[key] = (t,) + tuple(max([m for m in range(o + 1) if t[m] >= th] or [0]) for th in (.75, .5, .25))
    return r
def bt(t, need): return 1.0 if need <= 0 else (t[need] if need < len(t) else 0.0)
def prep(x):
    """Belief tails per face and the shortlist: per face the cheapest legal raise and the biggest claims that are >=75%, >=50%, >=25% likely."""
    c, last, total = x.c, x.last, x.n + x.others; lq, lf = (last // 6 + 1, last % 6 + 1) if last >= 0 else (0, 0)
    sums = [sum(fb) for _, fb in x.opp]; T = []; acts = []
    for f in range(6):
        h = hidden(tuple(sorted((nj, min(fb[f], 2), 1 if sums[j] > fb[f] else 0) for j, (nj, fb) in enumerate(x.opp)))); T.append(h[0])
        q0 = lq if (last >= 0 and f + 1 > lf) else lq + 1
        if q0 > total: continue
        qs = {q0}
        for nd in h[1:]:
            if q0 < c[f] + nd <= total: qs.add(c[f] + nd)
        for q in qs: acts.append((q - 1) * 6 + f)
    acts.sort()
    if last >= 0: acts.append(CALL)
    return T, acts
def share(cnts, i, dec=-1):
    """Estimated chance that player i wins the game, optionally after player `dec` loses a die."""
    p = [(n - (1 if j == dec else 0)) ** ALPHA if n - (1 if j == dec else 0) > 0 else 0.0 for j, n in enumerate(cnts)]; s = sum(p)
    return p[i] / s if s > 0 else 0.0
def clip3(v): return max(-3.0, min(3.0, v)) / 3
def feats3(x, T, acts):
    c, n, others, last, prev = x.c, x.n, x.others, x.last, x.prev; total = n + others; sd = math.sqrt(others * 5 / 36) or 1.0; mu = others / 6
    lq, lf = (last // 6 + 1, last % 6 + 1) if last >= 0 else (0, 0)
    cnts = [n] + [nj for nj, _ in x.opp]; v0 = share(cnts, 0); loss = (v0 - share(cnts, 0, 0)) * 4; gnext = (share(cnts, 0, 1) - v0) * 4
    if last >= 0:
        nl = lq - c[lf-1]
        st = [n / 5, others / 15, x.nopp / 3, 1.0, tail(others, nl), bt(T[lf-1], nl), clip3((nl - mu) / sd), (max(0, min(3, lq - (prev // 6 + 1))) / 3 if prev >= 0 else 0.0),
              min(x.nbids, 8) / 8, loss, (share(cnts, 0, len(cnts) - 1) - v0) * 4, gnext]
    else: st = [n / 5, others / 15, x.nopp / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, loss, 0.0, gnext]
    nfb = x.opp[0][1]; obf = [sum(fb[f] for _, fb in x.opp) for f in range(6)]; rows = []
    for a in acts:
        if a == CALL: rows.append(st + [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); continue
        q, f = a // 6 + 1, a % 6 + 1; nd = q - c[f-1]; q0 = lq if (last >= 0 and f > lf) else lq + 1
        rows.append(st + [0.0, tail(others, nd), bt(T[f-1], nd), clip3((nd - mu) / sd), max(0, min(4, q - lq)) / 4, 1.0 if (last >= 0 and f == lf) else 0.0, c[f-1] / n, f / 6, q / total,
                          min(obf[f-1], 3) / 3, 1.0 if nfb[f-1] else 0.0, 1.0 if q == q0 else 0.0])
    return np.array(rows, np.float32)
PL3, PA3 = (.03, .1, .2, .35, .5, .65, .8, .92), (.1, .25, .4, .55, .7, .85)
def tview(x, T, acts):
    """For the table agents: a situation key and one move per kind (call, or how likely x how big a jump x same face)."""
    c, last, others = x.c, x.last, x.others; lq, lf = (last // 6 + 1, last % 6 + 1) if last >= 0 else (0, 0)
    b = 0 if last < 0 else 1 + sum(bt(T[lf-1], lq - c[lf-1]) >= t for t in PL3)
    key = f"{b}|{0 if others <= 2 else 1 if others <= 5 else 2 if others <= 9 else 3}|{x.n}|{x.nopp}"; bm = {}; best = {}
    for a in acts:
        if a == CALL: bm[0] = a; continue
        q, f = a // 6 + 1, a % 6 + 1; pa = bt(T[f-1], q - c[f-1]); dq = q - lq
        k = 1 + sum(pa >= t for t in PA3) * 6 + (0 if dq <= 1 else 1 if dq == 2 else 2) * 2 + (1 if (last >= 0 and f == lf) else 0)
        if k not in bm or pa > best[k] + 1e-9: bm[k] = a; best[k] = pa
    return key, bm

# ---------- simulator ----------
def run_round(agents, cnt, hands, starter, on_decision=None):
    act = [s for s in range(len(cnt)) if cnt[s] > 0]; k = len(act); total = sum(cnt); fb = {s: [0] * 6 for s in act}; C = {s: counts(hands[s]) for s in act}
    last = prev = -1; i = act.index(starter); bidder = None; nb = 0
    while True:
        s = act[i]; x = X(); x.hand = hands[s]; x.c = C[s]; x.n = cnt[s]; x.others = total - cnt[s]; x.nopp = k - 1; x.last = last; x.prev = prev; x.nbids = nb
        x.opp = [(cnt[act[(i + d) % k]], fb[act[(i + d) % k]]) for d in range(1, k)]
        if on_decision: on_decision(s, x, [act[(i + d) % k] for d in range(1, k)], C)
        a = agents[s](x)
        if a == CALL:
            assert last >= 0
            return s if sum(C[t][last % 6] for t in act) >= last // 6 + 1 else bidder
        assert last < a < total * 6, (a, last, total)
        fb[s][a % 6] += 1; prev, last, bidder = last, a, s; nb += 1; i = (i + 1) % k
def play_game(agents, rng, n=4, start=5, on_round=None, on_decision=None):
    cnt = [start] * n; starter = rng.randrange(n)
    while sum(v > 0 for v in cnt) > 1:
        if on_round: on_round(cnt)
        hands = [G2.roll(v, rng) for v in cnt]; loser = run_round(agents, cnt, hands, starter, on_decision); cnt[loser] -= 1; starter = loser
        while cnt[starter] == 0: starter = (starter + 1) % n
    return next(s for s in range(n) if cnt[s] > 0)
def dv(cnt, loser):
    return [SCALE * (share(cnt, s, loser) - share(cnt, s)) for s in range(len(cnt))]
def train_round(choose, rng, pool, p_self=.5):
    cnt, hands = G2.deal(rng); k = len(cnt); role = [None if rng.random() < p_self else rng.choice(pool) for _ in range(k)]
    if all(r is not None for r in role): role[rng.randrange(k)] = None
    loser = run_round([(lambda x, s=s: choose(s, x)) if role[s] is None else role[s] for s in range(k)], cnt, hands, rng.randrange(k))
    return dv(cnt, loser), k

# ---------- agents ----------
def old(policy): return lambda x: policy((x.hand, x.n, x.others, x.nopp, x.last, x.prev))
math_agent = old(G2.math_bot)
def net_agent(net, greedy, nr=None):
    nr = nr or np.random.default_rng(9)
    def f(x):
        T, acts = prep(x)
        if len(acts) == 1: return acts[0]
        z = net.forward(feats3(x, T, acts))[:, 0]
        if greedy: return acts[int(z.argmax())]
        p = np.exp(z - z.max()); p /= p.sum(); return acts[int(nr.choice(len(acts), p=p))]
    return f
def table_agent(Tb, greedy, rng=None):
    rng = rng or random.Random(4)
    def f(x):
        T, acts = prep(x); key, bm = tview(x, T, acts); row = Tb.get(key)
        if row is None: return math_agent(x)
        bs = sorted(bm)
        if greedy: return bm[max(bs, key=lambda b: row[b])]
        w = [max(row[b], 0) for b in bs]
        return bm[rng.choices(bs, w)[0]] if sum(w) > 0 else math_agent(x)
    return f
def dec_net(layers):
    net = MLP([1]); net.W = []; net.b = []
    for L in layers:
        q = np.frombuffer(base64.b64decode(L['d']), '<i2').astype(np.float32) * L['s']; net.W.append(q[:L['i'] * L['o']].reshape(L['i'], L['o']).copy()); net.b.append(q[L['i'] * L['o']:].copy())
    return net
def gen1(name):
    """The previous generation (out5/), as an opponent."""
    d = json.load(open(f'out5/{name}.json'))
    if 'layers' in d: return old(G2.net_policy(dec_net(d['layers']), d['kind'] == 'q', np.random.default_rng(hash(name) % 1000)))
    return old(G2.table_policy(d['t'], d['kind'] == 'mcq', random.Random(hash(name) % 1000)))
def gen2(name, d=None):
    d = d or json.load(open(f'out6/{name}.json'))
    if 'layers' in d: return net_agent(dec_net(d['layers']), d['kind'] == 'q', np.random.default_rng(hash(name) % 1000 + 1))
    return table_agent(d['t'], d['kind'] == 'mcq', random.Random(hash(name) % 1000 + 1))
G1 = ('ppo', 'dqn', 'mc', 'nfsp', 'cfr')
def make_pool(phase):
    pool = [gen1(n) for n in G1] + [math_agent]
    if phase >= 2: pool += [gen2('p1_' + n) for n in G1 if os.path.exists(f'out6/p1_{n}.json')]
    return pool
_BENCH = None
def bench(agent, n=200, seed=0, start=5):
    """Win rate at a 4-player table against three bots drawn from the previous generation's best three. 25% = no better than them."""
    global _BENCH
    _BENCH = _BENCH or [gen1('dqn'), gen1('ppo'), gen1('mc')]
    rng = random.Random(seed); w = 0
    for g in range(n):
        ag = [rng.choice(_BENCH) for _ in range(4)]; ag[g % 4] = agent; w += play_game(ag, rng, 4, start) == g % 4
    return w / n
def point(curve, step, agent, t0):
    curve.append({"step": step, "vs_old": round(bench(agent), 3), "sec": round(time.time() - t0)}); print(curve[-1], flush=True)
    return curve[-1]["vs_old"] >= max(c["vs_old"] for c in curve)
def snap(net): c = MLP([1]); c.copy_from(net); return c
def save(name, obj, curve, phase):
    obj["curve"] = curve; json.dump(obj, open(f"out6/{'p1_' if phase == 1 else ''}{name}.json", "w"), separators=(",", ":"))
def prev_curve(name): return json.load(open(f'out6/p1_{name}.json'))['curve'] if os.path.exists(f'out6/p1_{name}.json') else []

# ---------- the five learners ----------
def train_mc(rounds=500_000, alpha=0.03, phase=1):
    rng = random.Random(1 + phase); pool = make_pool(phase); curve = []; t0 = time.time(); best = None
    Q = {k: np.array(v) for k, v in json.load(open('out6/p1_mc.json'))['t'].items()} if phase == 2 else {}
    for r in range(1, rounds + 1):
        eps = max(.05, (1 if phase == 1 else .3) * (1 - r / (rounds * .6))); steps = [[], [], [], []]
        def choose(s, x):
            T, acts = prep(x); key, bm = tview(x, T, acts); q = Q.get(key)
            if q is None: q = Q[key] = np.zeros(NB)
            bs = list(bm); b = rng.choice(bs) if rng.random() < eps else max(bs, key=lambda v: q[v]); steps[s].append((q, b)); return bm[b]
        rw, k = train_round(choose, rng, pool)
        for s in range(k):
            for q, b in steps[s]: q[b] += alpha * (rw[s] - q[b])
        if r % (rounds // 8) == 0 and point(curve, r, table_agent(Q, True), t0): best = {k: v.copy() for k, v in Q.items()}
    save("mc", {"kind": "mcq", "t": {k: [round(float(v), 3) for v in row] for k, row in best.items()}}, curve, phase)
def train_cfr(seconds=420, phase=1, league=0):
    """External-sampling MCCFR over the table abstraction. league=1: some seats are played by the crowd (earlier bots, Math Bot),
    so regret is measured against the opponents it will actually meet, not only against copies of itself."""
    rng = random.Random(phase + 10 * league); R = {}; curve = []; t0 = time.time(); best = None; it = 0; nxt = seconds / 8; pool = make_pool(phase) if league else []
    if phase == 2: R = {k: np.array(v) for k, v in json.load(open('out6/p1_cfr.json'))['r'].items()}
    def walk(hands, cnt, C, order, i, last, prev, bidder, fb, nb, upd, depth, roles):
        k = len(order); s = order[i]; x = X(); x.hand = hands[s]; x.c = C[s]; x.n = cnt[s]; x.others = sum(cnt) - cnt[s]; x.nopp = k - 1; x.last = last; x.prev = prev; x.nbids = nb
        x.opp = [(cnt[order[(i + d) % k]], fb[order[(i + d) % k]]) for d in range(1, k)]
        def val(a):
            if a == CALL: return dv(cnt, s if sum(C[t][last % 6] for t in order) >= last // 6 + 1 else bidder)[upd]
            fb[s][a % 6] += 1; v = walk(hands, cnt, C, order, (i + 1) % k, a, last, s, fb, nb + 1, upd, depth + (s == upd), roles); fb[s][a % 6] -= 1; return v
        if roles[s] is not None: return val(roles[s](x))
        T, acts = prep(x); key, bm = tview(x, T, acts); bs = sorted(bm); r = R.get(key)
        if r is None: r = R[key] = np.zeros(NB)
        pos = np.maximum(r[bs], 0); tt = pos.sum(); m = len(bs); pol = pos / tt if tt > 0 else np.full(m, 1 / m)
        if s == upd and depth < 2:
            vals = np.array([val(bm[b]) for b in bs]); v = float(pol @ vals); r[bs] += vals - v; return v
        return val(bm[bs[min(int(np.searchsorted(np.cumsum(pol), rng.random())), m - 1)]])
    while time.time() - t0 < seconds:
        it += 1; cnt, hands = G2.deal(rng); k = len(cnt); C = [counts(h) for h in hands]; order = list(range(k))
        roles = [rng.choice(pool) if (league and rng.random() < .5) else None for _ in range(k)]
        if all(r is not None for r in roles): roles[rng.randrange(k)] = None
        for upd in range(k):
            if roles[upd] is None: walk(hands, cnt, C, order, rng.randrange(k), -1, -1, None, [[0] * 6 for _ in range(k)], 0, upd, 0, roles)
        if time.time() - t0 >= nxt:
            nxt += seconds / 8; cur = {k: np.maximum(v, 0) for k, v in R.items()}
            if point(curve, it, table_agent(cur, False), t0): best = cur; bestR = {k: v.copy() for k, v in R.items()}
    save("cfr", {"kind": "cfrs", "t": {k: [round(float(v), 3) for v in (row / row.sum() if row.sum() > 0 else row)] for k, row in best.items()},
                 "r": {k: [round(float(v), 3) for v in row] for k, row in bestR.items()}}, curve, phase)
class Replay:
    def __init__(s, cap):
        s.cap, s.i, s.n = cap, 0, 0; s.f = np.zeros((cap, NF), np.float32); s.r = np.zeros(cap, np.float32)
        s.x2 = np.zeros((cap, MAXC, NF), np.float32); s.m2 = np.zeros((cap, MAXC), bool); s.d = np.zeros(cap, np.float32)
    def add(s, f, r, x2, d):
        i = s.i; s.f[i] = f; s.r[i] = r; s.d[i] = d; s.m2[i] = False
        if x2 is not None: s.x2[i, :len(x2)] = x2; s.m2[i, :len(x2)] = True
        s.i = (i + 1) % s.cap; s.n = min(s.n + 1, s.cap)
    def sample(s, b, nr): idx = nr.integers(0, s.n, b); return s.f[idx], s.r[idx], s.x2[idx], s.m2[idx], s.d[idx]
def push(buf, steps, rw, k):
    for s in range(k):
        tr = steps[s]
        for t, (F, ai) in enumerate(tr):
            if t + 1 < len(tr): buf.add(F[ai], 0.0, tr[t + 1][0], 0.0)
            else: buf.add(F[ai], rw[s], None, 1.0)
def dqn_update(q, qt, buf, nr, b=128):
    f, r, x2, m2, d = buf.sample(b, nr); flat = x2.reshape(-1, NF)
    a2 = np.where(m2, q.forward(flat).reshape(b, MAXC), -1e9).argmax(1)
    tgt = r + (1 - d) * qt.forward(flat).reshape(b, MAXC)[np.arange(b), a2]
    out = q.forward(f, cache=True); q.backward((np.clip(out[:, 0] - tgt, -1, 1) / b)[:, None])
def load_ws(name, keys):
    z = np.load(f'out6/ws_{name}.npz'); nets = []
    for kname in keys:
        n = MLP([1]); L = int(z[kname + '_n']); n.W = [z[f'{kname}_W{i}'] for i in range(L)]; n.b = [z[f'{kname}_b{i}'] for i in range(L)]
        m = MLP([w.shape[0] for w in n.W] + [n.W[-1].shape[1]], 0, 5e-4); m.copy_from(n); nets.append(m)
    return nets
def save_ws(name, **nets):
    d = {}
    for kname, n in nets.items():
        d[kname + '_n'] = len(n.W)
        for i, (w, b) in enumerate(zip(n.W, n.b)): d[f'{kname}_W{i}'] = w; d[f'{kname}_b{i}'] = b
    np.savez(f'out6/ws_{name}.npz', **d)
def train_dqn(rounds=90_000, phase=1):
    rng = random.Random(2 + phase); nr = np.random.default_rng(2 + phase); pool = make_pool(phase); curve = []; t0 = time.time(); best = None
    q = load_ws('dqn', ['q'])[0] if phase == 2 else MLP([NF, 64, 64, 1], 2, 1e-3); qt = snap(q); buf = Replay(40_000); ups = 0
    for r in range(1, rounds + 1):
        eps = max(.05, (1 if phase == 1 else .25) * (1 - r / (rounds * .5))); steps = [[], [], [], []]
        def choose(s, x):
            T, acts = prep(x); F = feats3(x, T, acts)
            ai = rng.randrange(len(acts)) if rng.random() < eps else int(q.forward(F)[:, 0].argmax()); steps[s].append((F, ai)); return acts[ai]
        rw, k = train_round(choose, rng, pool); push(buf, steps, rw, k)
        if buf.n > 3000:
            dqn_update(q, qt, buf, nr); ups += 1
            if ups % 400 == 0: qt.copy_from(q)
        if r % (rounds // 8) == 0 and point(curve, r, net_agent(q, True), t0): best = snap(q)
    save_ws('dqn', q=best); save("dqn", {"kind": "q", "layers": G2.export_net(best)}, curve, phase)
def train_ppo(iters=200, per=768, phase=1):
    rng = random.Random(3 + phase); nr = np.random.default_rng(3 + phase); pool = make_pool(phase); curve = []; t0 = time.time(); best = None
    pi, vf = load_ws('ppo', ['pi', 'vf']) if phase == 2 else (MLP([NF, 64, 64, 1], 3, 1e-3), MLP([12, 48, 48, 1], 4, 2e-3))
    for it in range(1, iters + 1):
        Xs, Mk, A, LP, Gr = [], [], [], [], []
        for _ in range(per):
            buf = []
            def choose(s, x):
                T, acts = prep(x); F = feats3(x, T, acts); z = pi.forward(F)[:, 0]; p = np.exp(z - z.max()); p /= p.sum()
                ai = int(nr.choice(len(acts), p=p)); buf.append((s, F, ai, math.log(p[ai] + 1e-12))); return acts[ai]
            rw, k = train_round(choose, rng, pool)
            for s, F, ai, lp in buf:
                xm = np.zeros((MAXC, NF), np.float32); xm[:len(F)] = F; m = np.zeros(MAXC, bool); m[:len(F)] = True
                Xs.append(xm); Mk.append(m); A.append(ai); LP.append(lp); Gr.append(rw[s])
        Xa, Mk, A, LP, Gr = np.array(Xs), np.array(Mk), np.array(A), np.array(LP), np.array(Gr, np.float32); N = len(Xa); ec = (.02 if phase == 1 else .008) * (1 - it / iters) + .002
        for _ in range(4):
            for idx in np.array_split(nr.permutation(N), max(1, N // 512)):
                xb, m, a, lp, g = Xa[idx], Mk[idx], A[idx], LP[idx], Gr[idx]; b = len(idx)
                v = vf.forward(xb[:, 0, :12], cache=True)[:, 0]; vf.backward(((v - g) / b)[:, None])
                adv = g - v; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                p = masked_softmax(pi.forward(xb.reshape(-1, NF), cache=True).reshape(b, MAXC), m); ratio = np.exp(np.log(p[np.arange(b), a] + 1e-12) - lp)
                use = ~(((ratio > 1.2) & (adv > 0)) | ((ratio < .8) & (adv < 0))); oh = np.zeros_like(p); oh[np.arange(b), a] = 1
                logp = np.log(p + 1e-12) * m; H = -(p * logp).sum(1, keepdims=True)
                pi.backward((((-(ratio * adv * use))[:, None] * (oh - p) + ec * p * (logp + H)) * m / b).reshape(-1, 1))
        if it % max(1, iters // 8) == 0 and point(curve, it * per, net_agent(pi, False), t0): best = snap(pi); bestv = snap(vf)
    save_ws('ppo', pi=best, vf=bestv); save("ppo", {"kind": "pi", "layers": G2.export_net(best)}, curve, phase)
def train_nfsp(rounds=60_000, eta=.25, phase=1):
    rng = random.Random(5 + phase); nr = np.random.default_rng(5 + phase); pool = make_pool(phase); curve = []; t0 = time.time(); best = None
    q, avg = load_ws('nfsp', ['q', 'avg']) if phase == 2 else (MLP([NF, 64, 64, 1], 5, 1e-3), MLP([NF, 64, 64, 1], 6, 1e-3)); qt = snap(q); buf = Replay(40_000)
    cap = 60_000; SX = np.zeros((cap, MAXC, NF), np.float32); SM = np.zeros((cap, MAXC), bool); SA = np.zeros(cap, np.int64); seen = 0; ups = 0
    for r in range(1, rounds + 1):
        mode = [rng.random() < eta for _ in range(4)]; steps = [[], [], [], []]
        def choose(s, x):
            nonlocal seen
            T, acts = prep(x); F = feats3(x, T, acts)
            if mode[s]:
                if rng.random() < .06: ai = rng.randrange(len(acts))
                else:
                    ai = int(q.forward(F)[:, 0].argmax()); j = seen if seen < cap else int(nr.integers(0, seen + 1))
                    if j < cap: SX[j] = 0; SX[j, :len(F)] = F; SM[j] = False; SM[j, :len(F)] = True; SA[j] = ai
                    seen += 1
            else:
                z = avg.forward(F)[:, 0]; p = np.exp(z - z.max()); p /= p.sum(); ai = int(nr.choice(len(acts), p=p))
            steps[s].append((F, ai)); return acts[ai]
        rw, k = train_round(choose, rng, pool); push(buf, steps, rw, k)
        if buf.n > 3000:
            dqn_update(q, qt, buf, nr); ups += 1
            if ups % 400 == 0: qt.copy_from(q)
        if seen > 1000:
            idx = nr.integers(0, min(seen, cap), 128); p = masked_softmax(avg.forward(SX[idx].reshape(-1, NF), cache=True).reshape(128, MAXC), SM[idx])
            oh = np.zeros_like(p); oh[np.arange(128), SA[idx]] = 1; avg.backward(((p - oh) * SM[idx] / 128).reshape(-1, 1))
        if r % (rounds // 8) == 0 and point(curve, r, net_agent(avg, False), t0): best = snap(avg); bestq = snap(q)
    save_ws('nfsp', q=bestq, avg=best); save("nfsp", {"kind": "pi", "layers": G2.export_net(best)}, curve, phase)

# ---------- fitting the two small models the bots share ----------
def fit(pop, games=1500, seed=0, start=5):
    """Play full games among `pop` and fit (a) alpha for the win-chance model and (b) the what-do-their-bids-mean tables."""
    rng = random.Random(seed); rounds = []; cnt_tab = {n: {b: {a: np.zeros(n + 1) for a in (0, 1)} for b in (0, 1, 2)} for n in range(1, 6)}
    def on_decision(s, x, others_seats, C):
        for (nj, fb), t in zip(x.opp, others_seats):
            sm = sum(fb)
            for f in range(6): cnt_tab[nj][min(fb[f], 2)][1 if sm > fb[f] else 0][C[t][f]] += 1
    for g in range(games):
        snaps = []; ag = [rng.choice(pop)] * 4 if g % 2 else [rng.choice(pop) for _ in range(4)]   # same-bot tables give the win-chance model without skill differences
        w = play_game(ag, rng, 4, start if g % 3 else 3, on_round=lambda c: snaps.append(list(c)), on_decision=on_decision)
        if g % 2 == 0: snaps = []
        rounds += [(c, w) for c in snaps]
    best = max((sum(math.log(max(1e-9, (c[w] ** a) / sum(v ** a for v in c if v > 0))) for c, w in rounds), a) for a in [x / 20 for x in range(10, 121)])
    post = {n: {b: {a: ((cnt_tab[n][b][a] + 30 * BIN[n]) / (cnt_tab[n][b][a].sum() + 30)).round(4).tolist() for a in (0, 1)} for b in (0, 1, 2)} for n in range(1, 6)}
    json.dump({"alpha": best[1], "post": post, "games": games, "samples": int(sum(cnt_tab[n][b][a].sum() for n in cnt_tab for b in (0, 1, 2) for a in (0, 1)))}, open('out6/model.json', 'w'))
    load_model(); return best[1], post
if __name__ == "__main__":
    if sys.argv[1] == 'fit':
        ph = int(sys.argv[2]); t0 = time.time(); a, post = fit(make_pool(ph), int(sys.argv[3]) if len(sys.argv) > 3 else 1500, seed=ph)
        print('alpha', a, '| holds of a face they bid once (5 dice):', post[5][1][0], '| never bid it but bid others:', post[5][0][1], '| prior:', BIN[5].round(4).tolist(), round(time.time() - t0), 's')
    else:
        kw = {k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])}; globals()["train_" + sys.argv[1]](**kw)
