"""Full-width CFR+ over all deals, vectorized with numpy.
Info set = (own hand, own dice, opp dice, last bid, previous bid), shared by both seats."""
import numpy as np, time, json, sys
import engine as E
from agents import curve_point, save, OBS, KEYS

def setup(a, b):
    HX, HY = E.HANDS[a], E.HANDS[b]
    w = np.outer([p for _, p in HX], [p for _, p in HY])
    T = a + b; B = 6 * T
    true = np.zeros((B, len(HX), len(HY)), bool)
    for k in range(B):
        q, f = E.bid_q(k), E.bid_f(k)
        true[k] = np.add.outer([h.count(f) for h, _ in HX], [h.count(f) for h, _ in HY]) >= q
    states = [(m, -1, -1) for m in (0,)] + [(m, l, p) for l in range(B) for p in range(-1, l) for m in (0, 1)]
    return dict(a=a, b=b, w=w, true=true, B=B, states=states)

TYPES = [setup(a, b) for a in (1, 2) for b in (1, 2)]
R, S = {}, {}
def tab(mn, on, l, p):
    k = (mn, on, l, p)
    if k not in R: R[k] = np.zeros((len(E.HANDS[mn]), E.NA)); S[k] = np.zeros((len(E.HANDS[mn]), E.NA))
    return k
def policy(k, l, T):
    acts = E.legal(l, T); r = R[k][:, acts]; s = r.sum(1, keepdims=True)
    p = np.where(s > 0, r / np.where(s > 0, s, 1), 1 / len(acts)); return acts, p

def iterate(t):
    for G in TYPES:
        a, b, w, true, B = G["a"], G["b"], G["w"], G["true"], G["B"]; T = a + b
        cnt = {0: (a, b), 1: (b, a)}
        pol, RX, RY, V = {}, {}, {}, {}
        for s in G["states"]:
            m, l, p = s; k = tab(*cnt[m], l, p); pol[s] = (k,) + policy(k, l, T)
        RX[(0, -1, -1)] = np.ones(len(E.HANDS[a])); RY[(0, -1, -1)] = np.ones(len(E.HANDS[b]))
        for s in G["states"]:                     # forward: counterfactual reaches (ordered by last bid)
            if s not in RX: continue
            m, l, p = s; k, acts, P = pol[s]
            for j, act in enumerate(acts):
                if act == E.CALL: continue
                c = (1 - m, act, l)
                rx = RX[s] * P[:, j] if m == 0 else RX[s]; ry = RY[s] * P[:, j] if m == 1 else RY[s]
                if c in RX: RX[c] = RX[c] + rx; RY[c] = RY[c] + ry
                else: RX[c] = rx; RY[c] = ry
        for s in reversed(G["states"]):           # backward: values (for seat X) + regret updates
            if s not in RX: continue
            m, l, p = s; k, acts, P = pol[s]
            Q = np.empty((len(acts),) + w.shape)
            for j, act in enumerate(acts):
                if act == E.CALL:
                    caller_loses = true[l]; loser_is_X = caller_loses if m == 0 else ~caller_loses
                    Q[j] = np.where(loser_is_X, -1.0, 1.0)
                else: Q[j] = V[(1 - m, act, l)]
            if m == 0:
                v = np.einsum('jxy,xj->xy', Q, P); V[s] = v
                reg = np.einsum('xy,y,jxy->xj', w, RY[s], Q - v[None]); own = RX[s]
            else:
                v = np.einsum('jxy,yj->xy', Q, P); V[s] = v
                reg = np.einsum('xy,x,jxy->yj', w, RX[s], -(Q - v[None])); own = RY[s]
            R[k][:, acts] = np.maximum(R[k][:, acts] + reg, 0)
            S[k][:, acts] += t * own[:, None] * P

def table():
    out = {}
    for o, key in zip(OBS, KEYS):
        hand, mn, on, l, p = o; acts = E.legal(l, mn + on); k = (mn, on, l, p); pr = np.zeros(E.NA)
        hi = [h for h, _ in E.HANDS[mn]].index(hand)
        if k in S and S[k][hi, acts].sum() > 0: pr[acts] = S[k][hi, acts] / S[k][hi, acts].sum()
        else: pr[acts] = 1 / len(acts)
        out[key] = pr
    return out

if __name__ == "__main__":
    iters = int(sys.argv[1]); curve = []; t0 = time.time()
    for t in range(1, iters + 1):
        iterate(t)
        if t in (1, 5, 10, 25) or t % 50 == 0: curve_point(curve, t, table(), t0)
    save("cfr", table(), curve, {"steps": "full sweeps", "infosets": len(R)})
