# Lab: training code, every generation of bots, and the exam

Working files behind the site. Nothing here is loaded by the web page.

## Bots, by stage

| Stage | Where | Notes |
|---|---|---|
| Original 2-dice bots | `site/ai/` | Trained only for four players with 2 dice each |
| First any-dice version | `site/ai5/` (as shipped), `out5/` (training output) | Trained by `gen.py` |
| Halfway through the retrain | `out6/p1_*.json`, `out6/model_p1.json`, `exam/bots_p1/` (browser format) | Phase 1 of `gen3.py` |
| Current | `site/bots/` (as shipped), `out6/*.json` (training output) | Phase 2 of `gen3.py`; `out6/ws_*.npz` are the warm-start weights |

`old_table.js` is the table script that shipped with the first two stages. It is kept so those bots can be run with their own code.

## Training

- `gen.py`: first any-dice generation.
- `gen3.py`: current generation (belief tables, game-aware reward, league training). `distill.py` makes the small network the Thinker uses.
- `export6.py`: writes the browser files. `league.js`: tournaments with the shipped JavaScript.
- `build2.py`, `make_texts.py`, `site/page.html`, `site/table_view.html`: build the page.
- Older scripts (`engine.py`, `agents.py`, `tournament*.py`, …) belong to the original 2-dice bots.

## The exam (`exam/`)

Two papers, both computed exactly rather than estimated from games.

**Paper A, the answer key.** One-on-one endgames with up to 3 dice on the table are solved exactly (`solver.c`, CFR+). Every situation a bot can face against the perfect player is graded against the best move (`grade.js`, `run_key.js`). The Thinker is a randomised search, so it is graded on sampled rounds (`run_thinker.js`).

**Paper B, the perfect hunter.** For one bot, `hunter.js` computes the best possible counter-strategy against that bot's actual move probabilities, in duels and at four-player tables where the other three seats are copies of the bot (`run_hunter.js`).

`lib.js` loads every stage with the JavaScript it shipped with and exposes each bot's full move distribution. `parity.js` checks those distributions against the shipped sampling code. `report.js` collects everything into `out/exam.json` and prints the tables. `explain.js` prints the hunter's plan against a bot in plain terms. `leaks.js` plays the two leaks shown on the site against the shipped bot code.

Results are in `exam/out/`: `exam.json` (everything), `summary.txt` (readable tables), and the raw `key_*.json` and `hunter_*.json` files.

To rebuild the answer keys (not stored here, 44 MB each):

```
cd exam && gcc -O3 -march=native -o solver solver.c -lm
./solver 1 1 0 20000 2e-7 2 out/nash_1_1_0.f32
./solver 2 1 0 4000 2e-5 14 out/nash_2_1_0.f32
./solver 2 1 1 4000 2e-5 14 out/nash_2_1_1.f32
node run_key.js 2 1 0 && node report.js
```

The paths in these scripts assume this folder layout (`lab/` as the working directory root).
