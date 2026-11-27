"""Assemble site/index.html: the four story chapters (site/page.html) + the table view (site/table_view.html) + results from out6/."""
import json, math, re, sys
sys.path.insert(0, '.')
L = lambda f: json.load(open(f))
NAMES = {"cfr": "Game Theorist", "dqn": "Exploiter", "ppo": "Gut Player", "nfsp": "Hybrid", "ismcts": "Thinker", "mc": "Old-Schooler", "math": "Math Bot", "random": "Random Bot"}
LEARN = ['ppo', 'dqn', 'mc', 'nfsp', 'cfr']; AIS = LEARN + ['ismcts']
strip = lambda d: {k.split(':')[1]: v for k, v in d.items()}
league = {}
for d in (5, 3, 2):
    j = L(f'out6/league_{d}.json'); league[d] = {"win": strip(j['win']), "style": strip(j['style'])}
vsold = L('out6/vsold_5.json')['win']; mixed = L('out6/mixed_5.json')['win']; duel = L('out6/duel_5.json')
# Elo from the one-on-one round robin (Bradley-Terry fit)
ids = list(NAMES); elo = {n: 0.0 for n in ids}
for _ in range(4000):
    for n in ids:
        g = 0
        for k, (w, m) in duel['res'].items():
            a, b = k.split('|')
            if n not in (a, b): continue
            o = b if n == a else a; s = w / m if n == a else 1 - w / m; g += m * (s - 1 / (1 + 10 ** ((elo[o] - elo[n]) / 400)))
        elo[n] += 0.01 * g
mean = sum(elo.values()) / len(elo); elo = {n: round(1500 + v - mean) for n, v in elo.items()}
topE = max(elo, key=elo.get); tw = tm = 0
for k, (w, m) in duel['res'].items():
    a, b = k.split('|')
    if topE == a: tw += w; tm += m
    elif topE == b: tw += m - w; tm += m
# training curves: phase 1 on the left half, phase 2 on the right
curves = {}; rounds = {}; secs = []
for n in LEARN:
    p1 = L(f'out6/p1_{n}.json')['curve']; p2 = L(f'out6/{n}.json')['curve']
    curves[n] = [[round((i + 1) / len(p1) * .5, 3), c['vs_old']] for i, c in enumerate(p1)] + [[round(.5 + (i + 1) / len(p2) * .5, 3), c['vs_old']] for i, c in enumerate(p2)]
    rounds[n] = p1[-1]['step'] + p2[-1]['step']; secs.append(p1[-1]['sec'] + p2[-1]['sec'])
big = lambda v: f"{v / 1e6:.1f} million" if v >= 1e6 else f"{round(v, -3):,}"
rtxt = {n: big(rounds[n]) + (" hands" if n == 'cfr' else " rounds") for n in LEARN}; rtxt['ismcts'] = "none"
model = L('out6/model.json'); post = model['post']; import gen3
gen3.load_model()
data = {"league": league, "vsold": vsold, "mixed": mixed, "duel": duel['res'], "elo": elo, "best1v1": round(tw / tm, 3), "curves": curves, "split": .5,
        "train": {"rounds": rtxt, "min": big(min(rounds.values())), "max": big(max(rounds.values())), "minutes": f"{round(min(secs) / 60)} to {round(max(secs) / 60)}"},
        "model": {"base2": round(float(sum(gen3.BIN[5][2:])), 3), "bid2": round(sum(post['5']['1']['0'][2:]), 3), "w55": .25, "w45": round(gen3.share([4, 5, 5, 5], 0), 3)}}
json.dump({'elo': elo, 'train': data['train'], 'alpha': model['alpha'], 'best1v1': data['best1v1'], 'topE': topE}, open('out6/data_cache.json', 'w'))
texts = L('out6/texts.json'); data.update(texts)
# ---- chapter 4: the exam (exam/out/exam.json, written by exam/report.js)
ex = L('exam/out/exam.json'); ES = ex['students']; MODES = ['table2', 'duel2', 'table1', 'duel3']; r0 = lambda v: math.floor(v + .5); pc = lambda v: f"{r0(v)}%"
def get(sid, f):
    try: return f(ES[sid])
    except (KeyError, TypeError): return None
stages = lambda n, f: [get(g + ':' + n, f) for g in ('old', 'mid', 'new')]   # first any-dice version, halfway, now
hunt = lambda m: (lambda s: [s['hunter'][m]['v'], s['hunter'][m]['err']])
XB = {}
for n in AIS:
    XB[n] = {"acc": stages(n, lambda s: s['key']['all']['paper']['acc']), "hard": stages(n, lambda s: s['key']['all']['hard']['acc']), "lost": stages(n, lambda s: s['key']['all']['paper']['lost']),
             "where": ES['new:' + n]['key']['all']['where'], "sampled": bool(ES['new:' + n]['key']['all']['sampled']), "hunter": {m: stages(n, hunt(m)) for m in MODES}}
ka = ES['math']['key']['all']
XB['math'] = {"acc": [None, None, ka['paper']['acc']], "hard": [None, None, ka['hard']['acc']], "lost": [None, None, ka['paper']['lost']], "where": ka['where'], "sampled": False, "hunter": {m: [None, None, get('math', hunt(m))] for m in MODES}}
orig = {}
for m in MODES:
    vs = [h[0] for h in (get('orig:' + n, hunt(m)) for n in LEARN) if h and h[1] < 5]
    orig[m] = [r0(min(vs)), r0(max(vs)), len(vs)] if vs else None
A = lambda n, k, i: XB[n][k][i]; T2 = lambda n, i: XB[n]['hunter']['table2'][i]
up = [n for n in LEARN if A(n, 'acc', 2) > A(n, 'acc', 0)]; easier = [n for n in LEARN if T2(n, 2)[0] > T2(n, 0)[0] + T2(n, 0)[1]]; harder = [n for n in LEARN if T2(n, 2)[0] + T2(n, 2)[1] < T2(n, 0)[0]]
# the wording below was written for these outcomes; fail loudly if a rerun changes them
assert sorted(up) == ['cfr', 'mc', 'nfsp', 'ppo'] and sorted(easier) == ['cfr', 'dqn', 'ppo'] and sorted(harder) == ['mc', 'nfsp'] and A('dqn', 'acc', 0) > 98 and XB['dqn']['hunter']['duel2'][0][0] > 90
take = [
 ["Retraining taught the hard decisions",
  f"On the questions where the raw odds mislead, the Game Theorist went from {pc(A('cfr', 'hard', 0))} to {pc(A('cfr', 'hard', 2))} and the Gut Player from {pc(A('ppo', 'hard', 0))} to {pc(A('ppo', 'hard', 2))}. "
  "Four of the five learners now score higher on the whole paper than their first version did. The Exploiter scores lower."],
 ["It did not make them harder to exploit",
  f"At a four-player table with 2 dice each, the hunter wins more against today's Gut Player ({pc(T2('ppo', 0)[0])} before, {pc(T2('ppo', 2)[0])} now), Exploiter ({pc(T2('dqn', 0)[0])}, {pc(T2('dqn', 2)[0])}) and Game Theorist ({pc(T2('cfr', 0)[0])}, {pc(T2('cfr', 2)[0])}) "
  "than against their first versions. Only the Old-Schooler and the Hybrid got harder to exploit."],
 ["A high mark is not the same as safe",
  f"The first Exploiter scored {pc(A('dqn', 'acc', 0))} on the answer key, yet the hunter beats it in {pc(XB['dqn']['hunter']['duel2'][0][0])} of duels. "
  "It plays good moves but always the same ones, so a counter-strategy can be built around them. That is why the exam needs both papers."]]
gapmax = max(v['gap'] for v in ex['keys'].values())
method = (f"Paper A covers one-on-one rounds with 1 die each and with 2 dice against 1. The answer key is within {gapmax * 100:.3f} points of perfect play, measured by what a best reply could still gain. "
          "A move counts as correct if it gives up at most half a point of round-win chance. Paper B is an exact best reply to each bot's move chances, playing whole games down from the starting dice. "
          "The hunter knows the bot's habits exactly, so its score is a worst case, not what a person at the table would get. "
          "As a spot check, several cases from each paper were also played out for 20,000 to 30,000 real games with the code that runs the table, and matched the computed figures.")
data['exam'] = {"situations": ex['situations'], "hardShare": round(ka['hard']['n'] / ka['paper']['n'], 3), "bots": XB, "orig": orig, "takeaways": take, "method": method}
json.dump(data['exam'], open('out6/exam_page.json', 'w'), indent=1)
page = open('site/page.html').read(); tv = open('site/table_view.html').read()
css = tv[tv.index('<style>') + 7: tv.index('</style>')]; html = tv[tv.index('</style>') + 8: tv.index('<script>')]
router = '''<script>
window.LD_WIN=%s;
{
const views={bots:document.getElementById("view-bots"),training:document.getElementById("view-training"),results:document.getElementById("view-results"),exam:document.getElementById("view-exam"),table:document.getElementById("view-table")};let loading=null,cur=null;
const load=src=>new Promise((ok,no)=>{const s=document.createElement("script");s.src=src;s.onload=ok;s.onerror=no;document.head.appendChild(s)});
function route(){const h=location.hash.slice(1),t=views[h]?h:"bots";for(const k in views)views[k].hidden=k!==t;
 document.querySelectorAll(".nav a").forEach(a=>a.setAttribute("aria-current",a.dataset.v===t?"page":"false"));
 if(t!==cur){cur=t;window.scrollTo(0,0)}
 if(t==="table"){if(window.LDTable)window.LDTable.resume();else if(!loading)loading=load("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js").then(()=>load("table.js")).catch(()=>{loading=null;document.getElementById("msg4").textContent="Couldn't load the 3D table. Check your connection and reopen this tab."})}
 else if(window.LDTable)window.LDTable.pause()}
addEventListener("hashchange",route);route();
}
</script>''' % json.dumps({d: league[d]['win'] for d in (2, 3, 5)}, separators=(',', ':'))
page = page.replace('<!--TABLECSS-->', css).replace('<!--TABLEHTML-->', html).replace('<!--TABLESCRIPT-->', router).replace('/*DATA*/', json.dumps(data, separators=(',', ':')))
open('site/index.html', 'w').write(page); print(len(page) // 1024, 'KB index.html | elo', elo, '| best 1v1', topE, data['best1v1'])
