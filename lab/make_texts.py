"""Write the result-dependent prose: out6/texts.json (for the page) and the repo README. All numbers come from out6/ result files."""
import json, sys, math
sys.path.insert(0, '.')
L = lambda f: json.load(open(f))
N = {"cfr": "Game Theorist", "dqn": "Exploiter", "ppo": "Gut Player", "nfsp": "Hybrid", "ismcts": "Thinker", "mc": "Old-Schooler", "math": "Math Bot", "random": "Random Bot"}
M = {"cfr": "CFR", "dqn": "DQN", "ppo": "PPO", "nfsp": "NFSP", "ismcts": "ISMCTS", "mc": "Monte Carlo", "math": "hand-written", "random": "hand-written"}
AIS = ['ppo', 'dqn', 'mc', 'nfsp', 'cfr', 'ismcts']
strip = lambda d: {k.split(':')[1]: v for k, v in d.items()}
pc = lambda v: f"{math.floor(v * 100 + .5 + 1e-9)}%"
lg = {d: L(f'out6/league_{d}.json') for d in (5, 3, 2)}; win = {d: strip(lg[d]['win']) for d in lg}; style = strip(lg[5]['style'])
vs = L('out6/vsold_5.json'); mx = L('out6/mixed_5.json'); du = L('out6/duel_5.json')
new = {k[4:]: v for k, v in mx['win'].items() if k.startswith('new:')}; old = {k[4:]: v for k, v in mx['win'].items() if k.startswith('old:')}
navg = sum(new.values()) / 6; oavg = sum(old.values()) / 6; gain = {n: new[n] - old[n] for n in AIS}; big = max(gain, key=gain.get); small = min(gain, key=gain.get)
better = [n for n in AIS if gain[n] > 0]; rank5 = sorted(win[5], key=win[5].get, reverse=True); top = rank5[0]
data = L('out6/data_cache.json'); elo = data['elo']; erank = sorted(elo, key=elo.get, reverse=True); topE = erank[0]
rk = {d: sorted(win[d], key=win[d].get, reverse=True) for d in (5, 3, 2)}
# the wording below was written for these outcomes; fail loudly if a rerun changes them
assert len(better) == 6 and big == 'ismcts' and small == 'cfr' and gain['cfr'] < .03 and all(rk[d][0] == 'ppo' for d in rk) and erank[:2] == ['ppo', 'dqn']
close = [n for n in AIS if n not in ('ppo', 'dqn') and abs(elo[n] - elo['math']) <= 25]
assert len(close) == 4
take = [
 ["The retraining worked",
  f"In one mixed field of all twelve bots, new and old, the new ones won {pc(navg)} of their games and the first versions {pc(oavg)}. "
  f"Every new bot finished ahead of its old self, though the {N['cfr']}'s edge ({pc(new['cfr'])} against {pc(old['cfr'])}) is too small to be sure of."],
 [f"The {N[big]} gained the most",
  f"It went from {pc(old[big])} to {pc(new[big])} in the mixed field, and wins {pc(vs['win'][big])} of its games against three copies of its old self. "
  "It now weighs the hidden dice by the bidding and plays its simulations out with a learned model of the others."],
 [f"The {N['ppo']} is the one to beat",
  f"It tops the four-player standings at 5, 3 and 2 dice ({pc(win[5]['ppo'])}, {pc(win[3]['ppo'])}, {pc(win[2]['ppo'])}) and has the best one-on-one rating. "
  f"Head to head the field is much tighter: only the {N['ppo']} and the {N['dqn']} are clearly ahead of the Math Bot. The learners practised at four-player tables, and it shows."]]
ci = lambda n, p=.25: round(196 * math.sqrt(p * (1 - p) / n))
npl = round(sum(lg[5]['played'].values()) / 8); nmx = round(sum(mx['played'].values()) / 12)
method = (f"Four-player standings: {lg[5]['games']:,} games per dice count, seats drawn at random from all eight players, so about {round(npl, -1):,} games per bot and a margin of roughly {ci(npl)} points either way. "
          f"Mixed field: {mx['games']:,} games, about {round(nmx, -1)} per bot (margin {ci(nmx)} points). New against old: {vs['games']} games per bot (margin {ci(vs['games'])} points). "
          f"One on one: {du['games']} games per pairing (margin {ci(du['games'], .5)} points per cell). "
          "Every game was played by the same JavaScript that runs the table. Training ran on one CPU core per bot with plain NumPy.")
oldnote = f"The line marks 25%. In a mixed field of all twelve bots, the new ones averaged {pc(navg)} and their first versions {pc(oavg)}."
json.dump({"takeaways": take, "method": method, "oldNote": oldnote}, open('out6/texts.json', 'w'), indent=1)
# ---------- README ----------
row = lambda n: f"| {N[n]} ({M[n]}) | " + " | ".join(("**" + pc(win[d][n]) + "**") if n == sorted(win[d], key=win[d].get, reverse=True)[0] else pc(win[d][n]) for d in (5, 3, 2)) + " |"
vrow = lambda n: f"| {N[n]} | {pc(old[n])} | {pc(new[n])} | {pc(vs['win'][n])} |"
tr = data['train']
# ---- the exam (out6/exam_page.json is written by build2.py)
X = L('out6/exam_page.json'); XB = X['bots']; XO = AIS + ['math']; xp = lambda v: '' if v is None else pc(v / 100)
hv = lambda c: '' if c is None else pc(c[0] / 100) + ('+' if c[1] >= 1 else '')
arrow = lambda cs: ' → '.join(hv(c) for c in cs if c is not None)
arow = lambda n: f"| {N[n]}{'*' if XB[n]['sampled'] else ''} | {xp(XB[n]['acc'][0])} | {xp(XB[n]['acc'][1])} | {xp(XB[n]['acc'][2])} | {xp(XB[n]['hard'][2])} |"
brow = lambda n: f"| {N[n]} | {arrow(XB[n]['hunter']['table2'])} | {arrow(XB[n]['hunter']['duel2'])} |"
hunted = [n for n in XO if XB[n]['hunter']['table2'][2]]
readme = f"""# Liar's Dice Invitational

Six AIs that each learned to bluff in a different way, a tournament between them, and a 3D table where you sit down and play them yourself.

> **Made for fun with [Claude Code](https://claude.com/claude-code).**
> This started as "I'm bored, what can we make?" The game, the six AIs, their training and this page all came out of that one conversation. It's a toy project, not research.

**Play it:** https://atharva309.github.io/Liars-dice/

![Your turn at the table: three AI opponents, your dice on a rack, and the coach suggesting a move](screenshots/your-turn.jpg)

![The reveal: cups lifted, matching dice counted, and the verdict](screenshots/reveal.jpg)

## What's here

The page reads as a short story in four chapters, followed by the game itself:

1. **The bots.** Who the eight players are and how each one thinks.
2. **Training.** The loop every learner ran, and what changed between the first version and the current one.
3. **Results.** Four-player standings, new bots against old, one-on-one records, and how each bot plays.
4. **Exam.** Two papers with right answers: an answer key for small endgames, and a "hunter" that finds each bot's worst-case opponent.
5. **Play.** A first-person 3D table against three AIs you pick, with 2, 3 or 5 dice each. Optional coach tips show the odds and suggest a move.

## The rules in 30 seconds

1. Everyone rolls their dice under a cup. You only see your own.
2. On your turn you make a claim about **all the dice on the table**, such as "at least 3 fives". Each claim must be higher than the last: more dice, or the same count with a higher number.
3. Instead of raising, you can call **Liar!** on the claim before yours. All cups lift and the dice are counted.
4. If the claim was true, the caller loses a die. If it was false, the claimer does. Nobody ever gains a die.
5. Lose all your dice and you're out. Last player with dice wins.

## The six AIs

| Name | Method | How it learns |
|---|---|---|
| Gut Player | PPO (policy gradient) | Makes the moves that came before good results a little more likely. It mixes its play, so it is hard to read. |
| Exploiter | DQN (deep Q-learning) | Learns a score for every move and always plays the highest. |
| Old-Schooler | Monte Carlo control | A lookup table of situations, nudged by how each round ended. |
| Hybrid | NFSP (neural fictitious self-play) | One network hunts for the best reply, a second averages its own past play. |
| Game Theorist | CFR (counterfactual regret minimization) | Tracks how much it regrets not having made each move. The family of methods behind the poker bots. |
| Thinker | ISMCTS-style search | No training of its own. Each turn it imagines the hidden dice and plays the round out many times, using a small learned model of how the others bid. |

Two hand-written bots fill out the field: the **Math Bot**, which always makes the most likely claim and calls when the odds say so, and the **Random Bot**.

## How they are trained

Nobody gave the bots a strategy. Each learner starts out making random claims and then repeats one loop a few hundred thousand times:

1. **Sit down with a crowd.** Each practice round seats the learner with copies of itself, the first version of every bot, and the Math Bot. A second phase adds the new bots as opponents for each other.
2. **Read the table.** The bot sees a shortlist of up to 24 claims and the chance each is true. That chance is adjusted by the bidding: a fitted table says how many of a face a player probably holds given what they have claimed so far.
3. **Get one number back.** When the round ends, the bot is told how much its estimated chance of winning the whole game moved. Losing a die costs more the fewer you have left.
4. **Adjust and repeat.** Each method uses that number differently, which is the only real difference between the bots.

Practice per bot ranged from {tr['min']} to {tr['max']} rounds, about {tr['minutes']} minutes each on one CPU core.

The bots were trained twice. What changed between the first version and the current one:

| | First version | Now |
|---|---|---|
| Opponents in training | Mostly copies of itself | Itself, every earlier bot, the Math Bot, then each other |
| Odds | Pure chance: every hidden die is 1 in 6 | Adjusted by what each player has bid |
| Rewarded for | Not losing a die this round | Raising its chance of winning the game |
| Options per turn | Up to 18 claims | Up to 24 claims |
| The Thinker | Assumes everyone plays the raw odds | Simulates with a learned model of the others |

## Results

Share of four-player games won, {lg[5]['games']:,} games per dice count with random seatings. An average player wins 25%.

| | 5 dice each | 3 dice | 2 dice |
|---|---|---|---|
""" + "\n".join(row(n) for n in rank5) + f"""

**New against old.** The middle columns come from one mixed field of all twelve bots ({mx['games']:,} games, 5 dice each). The last column seats each new bot with three copies of its first version ({vs['games']} games), where 25% would mean no change.

| | First version, mixed field | Current, mixed field | Current vs three of its first version |
|---|---|---|---|
""" + "\n".join(vrow(n) for n in sorted(AIS, key=new.get, reverse=True)) + f"""

On average the current bots won {pc(navg)} of their mixed-field games and the first versions {pc(oavg)}.

**One on one** (5 dice each, {du['games']} games per pairing, Elo fitted from the round robin): """ + ", ".join(f"{N[n]} {elo[n]}" for n in erank) + f""".

What stood out:

""" + "\n".join(f"- **{t[0]}.** {t[1]}" for t in take) + f"""

## The exam

Win rate against each other can hide habits that every bot shares, so the bots also sit an exam with right answers. Both papers are computed exactly instead of estimated from games.

**Paper A, the answer key.** One-on-one endgames with up to three dice on the table are solved completely: {X['situations'] / 1e6:.1f} million situations, each with a known best move. The table shows the share of a bot's moves that are the best move, or within half a point of it, for three versions of each bot.

| | First version | Halfway through retraining | Now | Hard questions, now |
|---|---|---|---|---|
""" + "\n".join(arow(n) for n in XO) + f"""

Hard questions are the {round(X['hardShare'] * 100)}% of decisions where playing the raw odds gives the wrong move, so the Math Bot scores 0% on them by definition. *The Thinker is a randomised search with no halfway version. It was marked on sampled rounds, so its figures are good to about 2 points.

**Paper B, the hunter.** For each bot, the best possible counter-strategy against that bot's actual habits. Nothing can beat the bot by more. Each cell shows how often the hunter wins against the first version → halfway → now. Lower is better for the bot.

| | Four players, 2 dice each (25% is an even share) | Duel, 2 dice each (50% means it cannot be exploited) |
|---|---|---|
""" + "\n".join(brow(n) for n in hunted) + f"""

A plus sign means "at least": that bot mixes its play too much to pin the figure down. The Thinker has no hunter score because its move chances cannot be written down exactly. The hunter knows a bot's habits exactly, so these are worst cases, not what a person at the table would get.

What the exam says:

""" + "\n".join(f"- **{t[0]}.** {t[1]}" for t in X['takeaways']) + f"""

Two leaks it found, both for when it is down to you and one bot with 2 dice each:

- **Exploiter.** Open with two of a low number you do not hold (ones, twos or threes). It always raises to three of that number, which cannot be true. Call it.
- **Gut Player.** When it makes the first claim of a round, it names a number it does not hold. Call Liar unless you hold that number yourself.

## How it works

- **Shortlist.** For each face a bot considers the cheapest legal raise and the biggest claims that are at least 75%, 50% and 25% likely, plus calling. That is at most 24 claims.
- **Reading the bidding.** A table fitted from the bots' own games gives the distribution of how many of a face a player holds, given how often they have bid that face this round. All probabilities are computed from it.
- **Playing for the game.** A player's chance of winning is modelled as their dice count to the power {data['alpha']}, as a share of everyone's. This was fitted from games and checked for calibration. The reward for a round is the change in that share.
- **Networks and tables.** Each option is described by 24 numbers. The neural bots score options with a 24-64-64-1 network. The table bots group situations into 600 kinds and moves into 43.
- **The Thinker.** It samples possible hands for the other players, weighted by their bids, tries each shortlisted move on the same sampled deals, and plays every round out with a small network distilled from the Gut Player.
- **Game Theorist.** External-sampling Monte Carlo CFR, from the same family of methods as the poker bot Pluribus, with some seats played by the crowd. It plays its final strategy: the long-run average, which is what the theory is about, tested much worse at four players.
- **Same code everywhere.** Training is plain NumPy. Every tournament game was played by the JavaScript this page ships, and that JavaScript is checked move for move against the training code.

## Run it locally

```
python3 -m http.server 8000
```

Then open http://localhost:8000. It needs a local server because the page fetches the bot files, and an internet connection for three.js and the fonts, which load from public CDNs.

## Files

- `index.html`: the three chapters, with results inlined
- `table.js`: the 3D table, the rules, and the code that runs the bots in the browser
- `bots/`: the trained bots and the two small models they share
- `screenshots/`: the images above

The training code, every earlier version of the bots, and the exam code are on the [`exam-lab` branch](https://github.com/Atharva309/Liars-dice/tree/exam-lab/lab) under `lab/`.

## Limitations

- These are small experiments with minutes of training on a CPU. The bots are fun opponents, not strong players by serious standards.
- Tournament results are bot against bot. How they fare against people is untested.
- The exam's answer key only covers one-on-one endgames with up to three dice on the table. Nobody has an answer key for the full game.
- The Game Theorist remains the weakest learner at a four-player table. CFR's guarantees only hold for two players.
- No wild ones and no "exact" call, so there is no way to win a die back.

---

Built with [Claude Code](https://claude.com/claude-code), for fun.
"""
open('out6/README.md', 'w').write(readme); print(json.dumps(take, indent=1)); print(method); print(oldnote)
