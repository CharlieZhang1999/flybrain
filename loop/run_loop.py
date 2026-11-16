"""Closed fly <-> LLM loop (Week 3 Day 5).

Brain: calibrated MaleCNS + presynaptic modulation (K = 0.05) + right-JO mirror
(cfg 'real'), or the same with the shuffled chemical matrix (cfg 'shuffle').

Each round, in one continuous simulation:
  1. 5 words back to back (500 ms each, 150 Hz)        -> word readout -> prompt
  2. LLM picks an action                                -> JO action code (500 ms)
  3. 500 ms of silence; descending-neuron readout decodes the action window
Each step resumes the simulator from its saved state (run_trial state= /
return_state=; spike-for-spike identical to one uninterrupted run, see
scripts/w3d6_resume_check.py), so brain time is simulated only once.
"""
import time
import numpy as np
from loop import words_readout, action_readout, prompt as P, llm
from loop.actions import ACTION_CODES_MIRROR, blocks as code_blocks

T_W, N_W, T_CODE, T_SIL, RATE, BIN = 0.5, 5, 0.5, 0.5, 150.0, 0.01
WORD_CFG = {'real': 'presyn0.05m', 'shuffle': 'presyn0.05ms'}
_BRAIN = {}


def brain(cfg):
    if cfg not in _BRAIN:
        from scripts.w3d1_common import presyn_model
        R, W, gap, ps, _ = presyn_model(0.05, 'calib' if cfg == 'real' else 'shuffle', mirror=True)
        _BRAIN[cfg] = (R, W, gap, ps)
    return _BRAIN[cfg]


def _sim(cfg, blocks, t_run, seed, state):
    from sim.lif_numpy import run_trial
    R, W, gap, ps = brain(cfg)
    p = dict(R.p); p['t_run'] = t_run
    r = run_trial(W, [], params=p, seed=seed, apl=R.apl, record=True, gap=gap, presyn=ps,
                  blocks=blocks, state=state, return_state=True)
    return r['t'] * p['dt'], r['i'], r['state']


def _counts(ts, si, a, b, n):
    return np.bincount(si[(ts >= a - 1e-9) & (ts < b - 1e-9)], minlength=n)


def run(seed, cfg='real', word_lists=None, rounds=3, provider=None):
    from scripts.ckpt2_ab import stim_idx
    from stimuli.malecns_words import WORDS
    R = brain(cfg)[0]; n = len(R.meta)
    wm, am = words_readout.load(WORD_CFG[cfg]), action_readout.load(cfg)
    rng = np.random.default_rng(seed)
    blocks, t, log, state = [], 0.0, [], None
    T, I = np.empty(0), np.empty(0, np.int64)
    t0 = time.time()
    for k in range(rounds):
        words = (word_lists[k] if word_lists and k < len(word_lists)
                 else [list(WORDS)[i] for i in rng.choice(len(WORDS), N_W, replace=False)])
        for j, w in enumerate(words):
            blocks.append((t + j * T_W, t + (j + 1) * T_W, stim_idx(R.meta, R.pos, w), RATE))
        t_words = t + N_W * T_W
        ts, si, state = _sim(cfg, blocks, t_words, seed, state)
        T, I = np.r_[T, ts], np.r_[I, si]; ts, si = T, I
        C = np.array([_counts(ts, si, t + j * T_W, t + (j + 1) * T_W, n) for j in range(N_W)])
        dec = words_readout.decode(wm, C)
        kept = [(w, pr) for w, pr in dec if w != 'none' and pr >= words_readout.UNSURE]
        text = P.build(kept)
        reply = llm.ask(text, provider=provider)
        action = reply['action']
        blocks += code_blocks(ACTION_CODES_MIRROR[action], R.meta, t0=t_words)
        t_end = t_words + T_CODE + T_SIL
        ts, si, state = _sim(cfg, blocks, t_end, seed, state)
        T, I = np.r_[T, ts], np.r_[I, si]; ts, si = T, I
        got, pa = action_readout.decode(am, _counts(ts, si, t_words, t_words + T_CODE, n))
        log.append(dict(round=k + 1, t_start=round(t, 2), true=words, decoded=[w for w, _ in dec],
                        p_top=[round(p_, 3) for _, p_ in dec],
                        words_correct=sum(a == b for a, b in zip(words, [w for w, _ in dec])),
                        prompt=text, llm_provider=reply['provider'], llm_model=reply['model'],
                        llm_raw=reply['raw'], action=action, code=ACTION_CODES_MIRROR[action],
                        action_decoded=got, action_p=round(pa, 3), action_ok=got == action,
                        firing_before_code=int(len(np.unique(si[(ts >= t_words - 0.1) & (ts < t_words)]))),
                        t_words=round(t_words, 2), t_end=round(t_end, 2)))
        t = t_end
    nb = int(round(t / BIN))
    pop = np.bincount((T / BIN).astype(int), minlength=nb)[:nb]
    spikes = dict(t=T, i=I)
    return dict(seed=seed, cfg=cfg, rounds=log, pop_10ms=pop.tolist(), t_total=t,
                wall_s=round(time.time() - t0, 1)), spikes
