"""Fly -> LLM episode (Week 3 Day 2).

play(): drive a list of words back to back (500 ms each, 150 Hz) in one
continuous simulation of the loop's brain model, then 500 ms of silence.
fly_to_llm(): play, decode each word window, build the prompt, ask the LLM.
"""
import time
import numpy as np
from loop import words_readout, prompt as P, llm

T_W, RATE, T_TAIL, BIN = 0.5, 150.0, 0.5, 0.01
_MODEL = {}


def brain(K=0.05):
    if 'brain' not in _MODEL:
        from scripts.w3d1_common import presyn_model
        R, W, gap, ps, _ = presyn_model(K)
        _MODEL['brain'] = (R, W, gap, ps)
    return _MODEL['brain']


def play(words, seed, K=0.05):
    from sim.lif_numpy import run_trial
    from scripts.ckpt2_ab import stim_idx
    R, W, gap, ps = brain(K)
    p = dict(R.p); p['t_run'] = len(words) * T_W + T_TAIL
    blocks = [(k * T_W, (k + 1) * T_W, stim_idx(R.meta, R.pos, w), RATE) for k, w in enumerate(words)]
    r = run_trial(W, [], params=p, seed=seed, apl=R.apl, record=True, gap=gap, blocks=blocks, presyn=ps)
    ts, si = r['t'] * p['dt'], r['i']
    n = len(R.meta)
    counts = np.array([np.bincount(si[(ts >= k * T_W) & (ts < (k + 1) * T_W)], minlength=n)
                       for k in range(len(words))])
    nb = int(round(p['t_run'] / BIN))
    pop = np.bincount((ts / BIN).astype(int), minlength=nb)[:nb]
    tail = len(np.unique(si[ts >= len(words) * T_W]))
    return counts, pop, tail


def fly_to_llm(words, seed, provider=None):
    t0 = time.time()
    counts, pop, tail = play(words, seed)
    dec = words_readout.decode(words_readout.load(), counts)
    kept = [(w, pr) for w, pr in dec if w != 'none' and pr >= words_readout.UNSURE]
    text = P.build(kept)
    reply = llm.ask(text, provider=provider)
    return dict(seed=seed, true=list(words), decoded=[w for w, _ in dec],
                p_top=[round(pr, 3) for _, pr in dec],
                unsure=[w for w, pr in dec if pr < words_readout.UNSURE],
                correct=sum(a == b for a, b in zip(words, [w for w, _ in dec])),
                prompt=text, system=P.SYSTEM, **{f'llm_{k}': v for k, v in reply.items()},
                tail_neurons=tail, pop_10ms=pop.tolist(), sim_s=round(time.time() - t0, 1))
