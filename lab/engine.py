"""Liar's Dice: 2 players, 2 dice each, 6 faces, no wilds.
A bid (q,f) = 'at least q dice show face f'. Raise = higher quantity, or same quantity + higher face.
Call 'liar' on the last bid: if true, caller loses a die; if false, bidder loses one.
Loser of a round starts the next. Lose all dice = lose the game."""
import itertools, random
from math import comb

FACES = 6
CALL = 24          # actions 0..23 are bids, 24 = call
NA = 25
def bid_q(k): return k // 6 + 1
def bid_f(k): return k % 6 + 1
def bid_str(k): return "LIAR!" if k == CALL else f"{bid_q(k)}x{bid_f(k)}"

def legal(last, total):
    acts = list(range(last + 1, total * 6))
    if last >= 0: acts.append(CALL)
    return acts

# all sorted hands of n dice with their probability
HANDS = {}
for n in (1, 2):
    cnt = {}
    for roll in itertools.product(range(1, 7), repeat=n):
        h = tuple(sorted(roll)); cnt[h] = cnt.get(h, 0) + 1
    HANDS[n] = [(h, c / 6 ** n) for h, c in sorted(cnt.items())]

def key(hand, mn, on, last, prev):
    return f"{''.join(map(str, hand))}|{mn}{on}|{last}|{prev}"

def all_obs():
    out = []
    for mn in (1, 2):
        for on in (1, 2):
            tot = mn + on
            for hand, _ in HANDS[mn]:
                out.append((hand, mn, on, -1, -1))
                for last in range(tot * 6):
                    for prev in range(-1, last):
                        out.append((hand, mn, on, last, prev))
    return out

def encode(obs):
    import numpy as np
    hand, mn, on, last, prev = obs
    x = np.zeros(60, dtype=np.float32)
    for d in hand: x[d - 1] += 0.5
    x[6 + mn - 1] = 1; x[8 + on - 1] = 1
    x[10 + last + 1] = 1; x[35 + prev + 1] = 1
    return x

def p_true(q, f, hand, on):
    """P(bid q x f is true) from one player's view, opponent has `on` unknown dice."""
    need = q - hand.count(f)
    if need <= 0: return 1.0
    if need > on: return 0.0
    return sum(comb(on, k) * (1/6)**k * (5/6)**(on-k) for k in range(need, on + 1))

def play_round(agents, counts, starter, rng, log=None):
    """agents: [fn(obs, legal, ctx)->action, ...]. Returns loser seat."""
    hands = [tuple(sorted(rng.randint(1, 6) for _ in range(counts[s]))) for s in (0, 1)]
    total = counts[0] + counts[1]
    last, prev, seat = -1, -1, starter
    history = []
    while True:
        obs = (hands[seat], counts[seat], counts[1 - seat], last, prev)
        acts = legal(last, total)
        a = agents[seat](obs, acts, {"history": history, "seat": seat})
        assert a in acts, (a, acts)
        if log is not None: log.append((seat, obs, a))
        if a == CALL:
            q, f = bid_q(last), bid_f(last)
            actual = hands[0].count(f) + hands[1].count(f)
            bidder = 1 - seat
            loser = seat if actual >= q else bidder
            if log is not None: log.append(("end", hands, loser))
            return loser
        history.append(a)
        prev, last, seat = last, a, 1 - seat

def play_game(a0, a1, starter, rng, log=None):
    counts = [2, 2]; agents = [a0, a1]
    while min(counts) > 0:
        loser = play_round(agents, counts, starter, rng, log)
        counts[loser] -= 1; starter = loser
    return 0 if counts[0] > 0 else 1

# ---------- baseline bots ----------
def random_bot(obs, acts, ctx, rng=random):
    # calls 1/3 of the time when it can, else random raise
    if CALL in acts and (len(acts) == 1 or rng.random() < 1/3): return CALL
    bids = [a for a in acts if a != CALL]
    return rng.choice(bids)

def math_bot(obs, acts, ctx=None):
    hand, mn, on, last, prev = obs
    bids = [a for a in acts if a != CALL]
    best, bp = None, -1
    for b in bids:
        p = p_true(bid_q(b), bid_f(b), hand, on)
        if p > bp + 1e-9: best, bp = b, p
    if last >= 0:
        pl = p_true(bid_q(last), bid_f(last), hand, on)
        if best is None or pl < 1 - bp: return CALL
    return best
