import json, random, sys, time, numpy as np
import engine as E
from nn import MLP, masked_softmax

OBS = E.all_obs()
X_ALL = np.stack([E.encode(o) for o in OBS])
M_ALL = np.zeros((len(OBS), E.NA), bool)
for i, o in enumerate(OBS):
    M_ALL[i, E.legal(o[3], o[1] + o[2])] = True
KEYS = [E.key(*o) for o in OBS]

def table_agent(table, rng=random):
    def act(obs, acts, ctx):
        p = table[E.key(*obs)]
        r = rng.random(); c = 0
        for a in acts:
            c += p[a]
            if r < c: return a
        return max(acts, key=lambda a: p[a])
    return act

def eval_vs(table, opp, n=1000, seed=0):
    rng = random.Random(seed); me = table_agent(table, rng)
    w = 0
    for g in range(n):
        if g % 2 == 0: w += E.play_game(me, opp, g // 2 % 2, rng) == 0
        else: w += E.play_game(opp, me, g // 2 % 2, rng) == 1
    return w / n

def curve_point(curve, step, table, t0):
    m = eval_vs(table, E.math_bot, 1000); r = eval_vs(table, lambda o, a, c: E.random_bot(o, a, c), 400, 1)
    curve.append({"step": step, "vs_math": round(m, 3), "vs_random": round(r, 3), "sec": round(time.time() - t0)})
    print(curve[-1], flush=True)

def save(name, table, curve, meta):
    tab = {k: [round(float(x), 3) for x in v] for k, v in table.items()}
    json.dump({"policy": tab, "curve": curve, "meta": meta}, open(f"out/{name}.json", "w"))

def net_table(net, greedy=False, kind="logits"):
    out = net.forward(X_ALL)
    if greedy:
        q = np.where(M_ALL, out, -1e9); P = np.zeros_like(out); P[np.arange(len(P)), q.argmax(1)] = 1
    else:
        P = masked_softmax(out, M_ALL)
    return dict(zip(KEYS, P))

def deal(rng):
    counts = [rng.choice((1, 2)), rng.choice((1, 2))]
    hands = [tuple(sorted(rng.randint(1, 6) for _ in range(counts[s]))) for s in (0, 1)]
    return counts, hands

def resolve(hands, last, caller):
    q, f = E.bid_q(last), E.bid_f(last)
    return caller if hands[0].count(f) + hands[1].count(f) >= q else 1 - caller

def selfplay_round(choose, rng):
    """choose(seat, obs, acts) -> action. Returns per-seat list of (obs, action), loser."""
    counts, hands = deal(rng); seat = rng.randint(0, 1)
    last = prev = -1; traj = ([], [])
    while True:
        obs = (hands[seat], counts[seat], counts[1-seat], last, prev)
        acts = E.legal(last, counts[0] + counts[1])
        a = choose(seat, obs, acts); traj[seat].append((obs, a))
        if a == E.CALL: return traj, resolve(hands, last, seat)
        prev, last, seat = last, a, 1 - seat

# ---------------- 1. CFR (outcome-sampling MCCFR) ----------------
def train_cfr(iters=1_500_000, eps=0.6):
    rng = random.Random(0); R, S = {}, {}; curve = []; t0 = time.time()
    def rm(r):
        p = np.maximum(r, 0); s = p.sum()
        return p / s if s > 0 else np.full(len(r), 1 / len(r))
    def ep(hands, counts, seat, last, prev, upd, my_r, opp_r, s_r):
        acts = E.legal(last, counts[0] + counts[1]); k = E.key(hands[seat], counts[seat], counts[1-seat], last, prev)
        if k not in R: R[k] = np.zeros(len(acts)); S[k] = np.zeros(len(acts))
        pol = rm(R[k]); n = len(acts)
        sp = eps / n + (1 - eps) * pol if seat == upd else pol
        ai = min(int(np.searchsorted(np.cumsum(sp), rng.random())), n - 1); a = acts[ai]
        if a == E.CALL:
            cv = 1.0 if resolve(hands, last, seat) != upd else -1.0
        else:
            cv = ep(hands, counts, 1 - seat, a, last, upd,
                    my_r * (pol[ai] if seat == upd else 1), opp_r * (pol[ai] if seat != upd else 1), s_r * sp[ai])
        child = np.zeros(n); child[ai] = cv / sp[ai]
        v = pol[ai] * child[ai]
        if seat == upd:
            R[k] += child * opp_r / s_r - v * opp_r / s_r
            np.maximum(R[k], 0, out=R[k])          # CFR+ regret floor
            S[k] += ep.t * my_r * pol / s_r         # linear averaging
        return v
    def table():
        t = {}
        for o, k in zip(OBS, KEYS):
            acts = E.legal(o[3], o[1] + o[2]); p = np.zeros(E.NA)
            if k in S and S[k].sum() > 0: p[acts] = S[k] / S[k].sum()
            else: p[acts] = 1 / len(acts)
            t[k] = p
        return t
    for it in range(1, iters + 1):
        counts, hands = deal(rng); ep.t = it
        ep(hands, counts, rng.randint(0, 1), -1, -1, it % 2, 1.0, 1.0, 1.0)
        if it % 500_000 == 0 or it == 50_000: curve_point(curve, it, table(), t0)
    save("cfr", table(), curve, {"steps": "iterations", "infosets": len(R)})

# ---------------- 2. Monte Carlo control ----------------
def train_mc(rounds=600_000, alpha=0.02):
    rng = random.Random(1); Q = {}; curve = []; t0 = time.time()
    def q(obs, acts):
        k = E.key(*obs)
        if k not in Q: Q[k] = {a: 0.0 for a in acts}
        return Q[k]
    def table():
        t = {}
        for o, k in zip(OBS, KEYS):
            acts = E.legal(o[3], o[1] + o[2]); p = np.zeros(E.NA)
            qq = Q.get(k); best = max(acts, key=lambda a: qq[a]) if qq else rng.choice(acts)
            p[best] = 1; t[k] = p
        return t
    for r in range(1, rounds + 1):
        epsi = max(0.05, 1 - r / (rounds * 0.6))
        def choose(seat, obs, acts):
            qq = q(obs, acts)
            return rng.choice(acts) if rng.random() < epsi else max(acts, key=lambda a: qq[a])
        traj, loser = selfplay_round(choose, rng)
        for s in (0, 1):
            G = -1.0 if s == loser else 1.0
            for obs, a in traj[s]:
                qq = Q[E.key(*obs)]; qq[a] += alpha * (G - qq[a])
        if r % 200_000 == 0 or r == 20_000: curve_point(curve, r, table(), t0)
    save("mc", table(), curve, {"steps": "rounds"})

# ---------------- shared DQN pieces ----------------
class Replay:
    def __init__(self, cap=100_000):
        self.cap = cap; self.i = 0; self.n = 0
        self.s = np.zeros((cap, 60), np.float32); self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, 60), np.float32); self.m2 = np.zeros((cap, E.NA), bool); self.d = np.zeros(cap, np.float32)
    def add(self, s, a, r, s2, m2, d):
        i = self.i; self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.m2[i] = m2; self.d[i] = d
        self.i = (i + 1) % self.cap; self.n = min(self.n + 1, self.cap)
    def sample(self, b, rng):
        idx = rng.integers(0, self.n, b); return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.m2[idx], self.d[idx]

def mask_of(acts):
    m = np.zeros(E.NA, bool); m[acts] = True; return m

def push_traj(buf, traj, loser):
    for s in (0, 1):
        G = -1.0 if s == loser else 1.0; tr = traj[s]
        for t, (obs, a) in enumerate(tr):
            x = E.encode(obs)
            if t + 1 < len(tr):
                o2 = tr[t+1][0]; buf.add(x, a, 0.0, E.encode(o2), mask_of(E.legal(o2[3], o2[1] + o2[2])), 0.0)
            else:
                buf.add(x, a, G, np.zeros(60, np.float32), np.ones(E.NA, bool), 1.0)

def dqn_update(q, qt, buf, nrng, b=128):
    s, a, r, s2, m2, d = buf.sample(b, nrng)
    q2 = q.forward(s2); a2 = np.where(m2, q2, -1e9).argmax(1)
    tgt = r + (1 - d) * qt.forward(s2)[np.arange(b), a2]
    out = q.forward(s, cache=True); g = np.zeros_like(out)
    err = out[np.arange(b), a] - tgt; g[np.arange(b), a] = np.clip(err, -1, 1) / b
    q.backward(g)

# ---------------- 3. DQN ----------------
def train_dqn(rounds=60_000):
    rng = random.Random(2); nrng = np.random.default_rng(2); curve = []; t0 = time.time()
    q = MLP([60, 128, 128, E.NA], 2, 5e-4); qt = MLP([60, 128, 128, E.NA], 2); qt.copy_from(q); buf = Replay(); ups = 0
    for r in range(1, rounds + 1):
        epsi = max(0.05, 1 - r / (rounds * 0.5))
        def choose(seat, obs, acts):
            if rng.random() < epsi: return rng.choice(acts)
            v = q.forward(E.encode(obs)[None])[0]; return max(acts, key=lambda a: v[a])
        traj, loser = selfplay_round(choose, rng); push_traj(buf, traj, loser)
        if buf.n > 2000:
            dqn_update(q, qt, buf, nrng); ups += 1
            if ups % 500 == 0: qt.copy_from(q)
        if r % 10_000 == 0 or r == 2000: curve_point(curve, r, net_table(q, greedy=True), t0)
    save("dqn", net_table(q, greedy=True), curve, {"steps": "rounds"})

# ---------------- 4. PPO ----------------
def train_ppo(iters=120, per=1024):
    rng = random.Random(3); nrng = np.random.default_rng(3); curve = []; t0 = time.time()
    pi = MLP([60, 128, 128, E.NA], 3, 3e-4); vf = MLP([60, 128, 128, 1], 4, 1e-3)
    for it in range(1, iters + 1):
        S, Mk, A, LP, G = [], [], [], [], []
        def choose(seat, obs, acts):
            m = mask_of(acts); p = masked_softmax(pi.forward(E.encode(obs)[None]), m[None])[0]
            a = int(nrng.choice(E.NA, p=p)); choose.buf.append((E.encode(obs), m, a, np.log(p[a] + 1e-12))); return a
        for _ in range(per):
            choose.buf = []; seats = []
            traj, loser = selfplay_round(lambda s, o, a: (seats.append(s), choose(s, o, a))[1], rng)
            for (x, m, a, lp), s in zip(choose.buf, seats):
                S.append(x); Mk.append(m); A.append(a); LP.append(lp); G.append(-1.0 if s == loser else 1.0)
        S, Mk, A, LP, G = map(np.array, (S, Mk, A, LP, G)); N = len(S)
        ent_c = 0.02 * (1 - it / iters) + 0.002
        for _ in range(4):
            for idx in np.array_split(nrng.permutation(N), max(1, N // 512)):
                s, m, a, lp, g = S[idx], Mk[idx], A[idx], LP[idx], G[idx]; b = len(idx)
                v = vf.forward(s, cache=True)[:, 0]; vf.backward(((v - g) / b)[:, None])
                adv = g - v; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                logits = pi.forward(s, cache=True); p = masked_softmax(logits, m)
                ratio = np.exp(np.log(p[np.arange(b), a] + 1e-12) - lp)
                use = ~(((ratio > 1.2) & (adv > 0)) | ((ratio < 0.8) & (adv < 0)))
                # d(-ratio*adv)/dlogits = -ratio*adv*(onehot - p)
                oh = np.zeros_like(p); oh[np.arange(b), a] = 1
                g_pol = (-(ratio * adv * use))[:, None] * (oh - p)
                logp = np.log(p + 1e-12) * m; H = -(p * logp).sum(1, keepdims=True)
                g_ent = ent_c * p * (logp + H)          # gradient of -entropy
                pi.backward(((g_pol + g_ent) * m) / b)
        if it % 25 == 0 or it == 2: curve_point(curve, it * per, net_table(pi), t0)
    save("ppo", net_table(pi), curve, {"steps": "rounds"})

# ---------------- 5. NFSP ----------------
def train_nfsp(rounds=80_000, eta=0.25):
    rng = random.Random(5); nrng = np.random.default_rng(5); curve = []; t0 = time.time()
    q = MLP([60, 128, 128, E.NA], 5, 5e-4); qt = MLP([60, 128, 128, E.NA], 5); qt.copy_from(q)
    avg = MLP([60, 128, 128, E.NA], 6, 5e-4); buf = Replay()
    cap = 200_000; SLs = np.zeros((cap, 60), np.float32); SLm = np.zeros((cap, E.NA), bool); SLa = np.zeros(cap, np.int64); seen = 0; ups = 0
    for r in range(1, rounds + 1):
        epsi = 0.06; mode = [rng.random() < eta, rng.random() < eta]
        def choose(seat, obs, acts):
            nonlocal seen
            x = E.encode(obs); m = mask_of(acts)
            if mode[seat]:
                if rng.random() < epsi: return rng.choice(acts)   # exploration not copied into average
                v = q.forward(x[None])[0]; a = max(acts, key=lambda k: v[k])
                j = seen if seen < cap else nrng.integers(0, seen + 1)
                if j < cap: SLs[j] = x; SLm[j] = m; SLa[j] = a
                seen += 1; return a
            p = masked_softmax(avg.forward(x[None]), m[None])[0]; return int(nrng.choice(E.NA, p=p))
        traj, loser = selfplay_round(choose, rng); push_traj(buf, traj, loser)
        if buf.n > 2000:
            dqn_update(q, qt, buf, nrng); ups += 1
            if ups % 500 == 0: qt.copy_from(q)
        if seen > 1000:
            idx = nrng.integers(0, min(seen, cap), 128)
            logits = avg.forward(SLs[idx], cache=True); p = masked_softmax(logits, SLm[idx])
            oh = np.zeros_like(p); oh[np.arange(128), SLa[idx]] = 1; avg.backward((p - oh) * SLm[idx] / 128)
        if r % 20_000 == 0 or r == 2000: curve_point(curve, r, net_table(avg), t0)
    save("nfsp", net_table(avg), curve, {"steps": "rounds"})

if __name__ == "__main__":
    name = sys.argv[1]; kw = {k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])}
    globals()["train_" + name](**kw)
