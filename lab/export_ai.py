"""Write one small file per agent so the page only loads the AIs that sit down.
Nets are stored as 16-bit integers (base64) instead of JSON numbers: ~7x smaller."""
import json, base64, os, numpy as np
os.makedirs('site/ai', exist_ok=True)
for n in ('cfr', 'mc', 'dqn', 'ppo', 'nfsp'):
    d = json.load(open(f'out4/{n}.json'))
    if d['kind'] == 'table':
        if n == 'mc': out = {'kind': 'det', 'policy': {k: v[0][0] for k, v in d['policy'].items()}}
        else: out = {'kind': 'table', 'policy': {k: [[a, round(p, 2)] for a, p in v if p >= 0.01] for k, v in d['policy'].items()}}
    else:
        layers = []
        for W, b in zip(d['policy']['W'], d['policy']['b']):
            W = np.array(W, np.float32); b = np.array(b, np.float32)
            s = float(max(np.abs(W).max(), np.abs(b).max())) / 32767
            q = np.round(np.concatenate([W.ravel(), b]) / s).astype('<i2')
            layers.append({'i': W.shape[0], 'o': W.shape[1], 's': s, 'd': base64.b64encode(q.tobytes()).decode()})
        out = {'kind': d['kind'], 'layers': layers}
    json.dump(out, open(f'site/ai/{n}.json', 'w'), separators=(',', ':'))
    print(n, out['kind'], os.path.getsize(f'site/ai/{n}.json') // 1024, 'KB')
