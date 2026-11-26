# Liar's Dice Invitational

Six AIs that each learned Liar's Dice a different way (CFR, DQN, PPO, NFSP, Monte Carlo control, ISMCTS), a tournament between them, and a 3D table where you play against them with 2, 3 or 5 dice each.

Static site, no build step: `index.html`, `table.js`, and the trained agents in `ai/` (2-dice game) and `ai5/` (3- and 5-dice games). Three.js and the fonts load from public CDNs.

## Publish with GitHub Pages
1. Put these files at the root of a repository (keep the `ai` and `ai5` folders).
2. Repository **Settings → Pages → Build and deployment**: Source "Deploy from a branch", branch `main`, folder `/ (root)`.
3. After a minute the site is at `https://<username>.github.io/<repository>/`.
