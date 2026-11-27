"""The Thinker needs a fast stand-in for 'how do good players bid?'. Distil the Gut Player (PPO) into a much smaller network
by showing it the situations that come up in league games and training it to copy the Gut Player's move probabilities."""
import json, random, sys, time, numpy as np
import gen3 as M, gen as G2
from nn import MLP, masked_softmax
t0 = time.time(); rng = random.Random(7); nr = np.random.default_rng(7)
teacher = M.dec_net(json.load(open('out6/ppo.json'))['layers'])
names = [n for n in M.G1 if __import__('os').path.exists(f'out6/{n}.json')]
pop = [M.gen2(n) for n in names] + [M.gen2('p1_' + n) for n in M.G1] + [M.gen1('dqn'), M.gen1('ppo'), M.math_agent]
X, Mk, P = [], [], []
def on_decision(s, x, others, C):
    T, acts = M.prep(x)
    if len(acts) < 2 or rng.random() > .5: return
    F = M.feats3(x, T, acts); z = teacher.forward(F)[:, 0]; p = np.exp(z - z.max()); p /= p.sum()
    xm = np.zeros((M.MAXC, M.NF), np.float32); xm[:len(F)] = F; m = np.zeros(M.MAXC, bool); m[:len(F)] = True; pp = np.zeros(M.MAXC, np.float32); pp[:len(F)] = p
    X.append(xm); Mk.append(m); P.append(pp)
g = 0
while len(X) < 90_000:
    g += 1; M.play_game([rng.choice(pop) for _ in range(rng.choice((2, 3, 4, 4)))], rng, n=None or 4, start=rng.choice((2, 3, 5, 5)), on_decision=on_decision) if False else None
    k = rng.choice((2, 3, 4, 4)); M.play_game([rng.choice(pop) for _ in range(k)], rng, k, rng.choice((2, 3, 5, 5)), on_decision=on_decision)
X, Mk, P = np.array(X), np.array(Mk), np.array(P); N = len(X); print(N, 'situations from', g, 'games', round(time.time() - t0), 's', flush=True)
H = int(sys.argv[1]) if len(sys.argv) > 1 else 24
st = MLP([M.NF, H, 1], 11, 3e-3); hold = slice(0, 4000); tr = np.arange(4000, N)
for ep in range(24):
    if ep == 16: st.lr = 1e-3
    for idx in np.array_split(nr.permutation(tr), len(tr) // 256):
        b = len(idx); p = masked_softmax(st.forward(X[idx].reshape(-1, M.NF), cache=True).reshape(b, M.MAXC), Mk[idx]); st.backward(((p - P[idx]) * Mk[idx] / b).reshape(-1, 1))
    if ep % 4 == 3:
        p = masked_softmax(st.forward(X[hold].reshape(-1, M.NF)).reshape(4000, M.MAXC), Mk[hold]); q = P[hold]
        kl = float((q * (np.log(q + 1e-9) - np.log(p + 1e-9)) * Mk[hold]).sum(1).mean()); print('epoch', ep + 1, 'KL', round(kl, 4), 'same top move', round(float((p.argmax(1) == q.argmax(1)).mean()), 3), flush=True)
print('lite vs old top three:', M.bench(M.net_agent(st, False), 300), '| teacher:', M.bench(M.net_agent(teacher, False), 300), round(time.time() - t0), 's')
json.dump({"kind": "pi", "layers": G2.export_net(st)}, open('out6/lite.json', 'w'), separators=(',', ':'))
