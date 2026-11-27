"""Assemble site/index.html from the tournament template + the table view. Data is inlined; table.js and ai/*.json sit beside it."""
import json
s = open('site/template.html').read()
def rep(a, b, n=1):
    global s
    assert a in s, a[:60]; s = s.replace(a, b, n)
# drop the old 2D game (section + code)
a = s.index('<section class="band" style="padding-top:8px" id="play">'); b = s.index('</section>', a) + len('</section>'); s = s[:a] + s[b:]
a = s.index('// ================= GAME ENGINE'); b = s.index('\n}\n</script>', a); s = s[:a] + s[b:]
rep('<header class="band">', '<nav class="nav" aria-label="Views"><a href="#" data-v="tour">Tournament</a><a href="#table" data-v="table">Play at the table</a></nav>\n<div id="view-tour">\n<header class="band">')
rep("Six AIs that each learned to bluff in a different way, plus two warm-up bots. They played every opponent thousands of times. Then it's your turn at the table.",
    "Six AIs that each learned to bluff in a different way, plus two warm-up bots. They played each other thousands of times, one-on-one and four to a table. Then it's your turn.")
rep('<div class="champ" id="champ"></div>', '<a class="cta" href="#table">Play at the table</a><div class="champ" id="champ"></div>')
# standings: one-on-one beside four-player
rep('''<section class="band alt"><div class="wrap">
<h2 id="lbTitle">Final standings</h2>
<p class="sub">Elo rating fitted from every game of the round robin. A 100-point gap means the higher-rated bot wins about 64% of games.</p>
<div class="lb" id="lb"></div>
</div></section>''', '''<section class="band alt"><div class="wrap grid2">
<div><h2>One-on-one standings</h2>
<p class="sub">Elo rating fitted from every game of the round robin. A 100-point gap means the higher-rated bot wins about 64% of games.</p>
<div class="lb" id="lb"></div></div>
<div><h2>Four at a table</h2>
<p class="sub">Share of four-player games won, with random seatings. An average player wins 25%. For 3 and 5 dice each the AIs were retrained to choose from a shortlist of sensible claims using the odds of each.</p>
<div class="seg2" id="seg4"><button data-m="2">2 dice each</button><button data-m="3">3 dice</button><button data-m="5">5 dice</button></div>
<div class="lb" id="lb4"></div></div>
</div></section>''')
rep('''document.getElementById("champ").textContent=`Champion: ${N[ranked[0]]} (${INFO[ranked[0]][0]}), Elo ${T.elo[ranked[0]]}`;''',
    '''const WINS=window.LD_WIN,rank4=m=>Object.keys(WINS[m]).sort((a,b)=>WINS[m][b]-WINS[m][a]);let mode4=5;
const draw4=()=>{const W4=WINS[mode4],top=Math.max(.45,...Object.values(W4));document.getElementById("lb4").innerHTML=rank4(mode4).map((n,i)=>`<div class="row"><span class="rk">${i+1}</span><span class="nm"><b>${esc(N[n])}</b><span>${esc(n==="cfr"&&mode4>2?"Sampled CFR (Pluribus-style)":INFO[n][0])}</span></span><div class="bar"><i style="width:${(W4[n]/top*100).toFixed(1)}%;background:${COL[n]}"></i></div><span class="elo">${Math.round(W4[n]*100)}%</span></div>`).join('');
 document.querySelectorAll("#seg4 button").forEach(b=>b.setAttribute("aria-pressed",+b.dataset.m===mode4))};
document.querySelectorAll("#seg4 button").forEach(b=>b.onclick=()=>{mode4=+b.dataset.m;draw4()});draw4();
document.getElementById("champ").textContent=`One-on-one: ${N[ranked[0]]} · Four-player, 5 dice: ${N[rank4(5)[0]]}`;''')
rep('.row{display:grid;grid-template-columns:2.2em minmax(0,13em) minmax(0,1fr) 4.2em;', '.seg2{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 16px}.seg2 button{background:var(--card);color:var(--ink);border:1px solid var(--line)}.seg2 button[aria-pressed=true]{background:var(--ink);color:var(--bg);border-color:var(--ink)}\n.row{display:grid;grid-template-columns:2.2em minmax(0,10.5em) minmax(0,1fr) 3.6em;')
rep('<p class="note">Trained by self-play on a 2-dice-each game. The Thinker', '<p class="note">These personality numbers come from the one-on-one tournament. Trained by self-play on a 2-dice-each game. The Thinker')
t = json.load(open('out/tournament.json')); ag = json.load(open('site/agents.json')); t4 = json.load(open('out4/tournament4.json'))
wins = {2: t4['win'], 3: json.load(open('out5/tour_3.json'))['win'], 5: json.load(open('out5/tour_5.json'))['win']}
rep('/*TOURNAMENT*/', json.dumps(t, separators=(',', ':'))); rep('/*CURVES*/', json.dumps(ag['curves'], separators=(',', ':')))
part = open('site/table_view.html').read().replace('/*T4*/', json.dumps(wins, separators=(',', ':')))
css, rest = part.split('</style>', 1)
rep('</style></head>', css.replace('<style>', '') + '</style></head>')
rep('<script>\nconst T=', '</div>\n' + rest + '\n<script>\nconst T=')
open('site/index.html', 'w').write(s); print(len(s) // 1024, 'KB index.html')
