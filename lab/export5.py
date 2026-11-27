import json, os
os.makedirs('site/ai5', exist_ok=True)
for n in ('cfr', 'mc', 'dqn', 'ppo', 'nfsp'):
    if not os.path.exists(f'out5/{n}.json'): print(n, 'not ready'); continue
    d = json.load(open(f'out5/{n}.json')); d.pop('curve', None)
    if 't' in d: d['t'] = {k: [round(x, 2) for x in v] for k, v in d['t'].items()}
    json.dump(d, open(f'site/ai5/{n}.json', 'w'), separators=(',', ':')); print(n, d['kind'], os.path.getsize(f'site/ai5/{n}.json') // 1024, 'KB')
