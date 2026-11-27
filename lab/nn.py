"""Tiny numpy MLP with Adam."""
import numpy as np
class MLP:
    def __init__(self, sizes, seed=0, lr=1e-3):
        r = np.random.default_rng(seed)
        self.W = [r.normal(0, np.sqrt(2/a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.lr = lr; self.t = 0
        self.m = [np.zeros_like(p) for p in self.W + self.b]; self.v = [np.zeros_like(p) for p in self.W + self.b]
    def forward(self, x, cache=False):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1: x = np.maximum(x, 0)
            hs.append(x)
        if cache: self._hs = hs
        return x
    def backward(self, dout):
        hs = self._hs; gW, gb = [], []
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1: dout = dout * (hs[i+1] > 0)
            gW.insert(0, hs[i].T @ dout); gb.insert(0, dout.sum(0))
            dout = dout @ self.W[i].T
        self.step(gW + gb)
    def step(self, grads, b1=0.9, b2=0.999):
        self.t += 1; ps = self.W + self.b
        for p, g, m, v in zip(ps, grads, self.m, self.v):
            np.clip(g, -5, 5, out=g)
            m *= b1; m += (1-b1)*g; v *= b2; v += (1-b2)*g*g
            p -= self.lr * (m/(1-b1**self.t)) / (np.sqrt(v/(1-b2**self.t)) + 1e-8)
    def copy_from(self, o):
        self.W = [w.copy() for w in o.W]; self.b = [b.copy() for b in o.b]
    def export(self):
        return {"W": [np.round(w, 4).tolist() for w in self.W], "b": [np.round(b, 4).tolist() for b in self.b]}

def masked_softmax(logits, mask):
    z = np.where(mask, logits, -1e9); z = z - z.max(-1, keepdims=True)
    e = np.exp(z) * mask; return e / e.sum(-1, keepdims=True)
