"""Write the new-generation bots for the browser: site/bots/. Usage: python3 export6.py [p1]  (p1 = export the phase-1 snapshots for testing)"""
import json, os, sys
pre = 'p1_' if len(sys.argv) > 1 and sys.argv[1] == 'p1' else ''
os.makedirs('site/bots', exist_ok=True)
TEMP = {'ppo': .5, 'nfsp': .4, 'lite': .7}   # sampled policies are sharpened at play time (probabilities ^ 1/temp)
m = json.load(open('out6/model.json')); json.dump({"alpha": m["alpha"], "post": m["post"], "rollout": "lite"}, open('site/bots/model.json', 'w'), separators=(',', ':'))
for n in ('cfr', 'mc', 'dqn', 'ppo', 'nfsp', 'lite'):
    src = f'out6/{n}.json' if n == 'lite' else f'out6/{pre}{n}.json'
    if not os.path.exists(src): print(n, 'not ready'); continue
    d = json.load(open(src)); d.pop('curve', None); d.pop('r', None)
    if n in TEMP: d['temp'] = TEMP[n]
    if 't' in d: d['t'] = {k: [round(v, 3) for v in row] for k, row in d['t'].items()}
    json.dump(d, open(f'site/bots/{n}.json', 'w'), separators=(',', ':')); print(n, d['kind'], os.path.getsize(f'site/bots/{n}.json') // 1024, 'KB')
