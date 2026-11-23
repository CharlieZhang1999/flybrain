"""scripts/w5_hidden_state.py: episode design and label hiding (no data/, no torch)."""
import json

from scripts import w5_hidden_state as h
from stimuli import states as st


def test_segments_design():
    seen = set()
    for k in range(h.N_EP):
        segs = h.segments_for(k)
        assert segs == h.segments_for(k)                         # depends only on k
        assert [(s['t_on'], s['t_off']) for s in segs] == [(0.0, 2.0), (2.0, 4.0)]
        for s in segs:
            assert s['state'] in h.STATES_USED
            lo, hi = (x * h.LEVEL_HZ[s['state']] for x in h.LEVEL_SCALE)
            assert lo - 1e-3 <= s['level_hz'] <= hi + 1e-3
            seen.add(s['state'])
        sid, lev = st.label_bins(segs, h.T, h.LABEL_BIN_S)
        assert len(sid) == 400 and set(sid[:200]) == {st.STATE_ID[segs[0]['state']]}
    assert seen == set(h.STATES_USED)                            # sleep (ER5) never driven


def test_words_spec_has_no_state():
    for k in (0, 1, h.N_EP - 1):
        sp = h.spec_for(k)
        assert sp['T'] == h.T and sp['spec_seed'] == h.SPEC_SEED0 + k
        txt = json.dumps(sp)
        assert 'state' not in txt and 'level' not in txt
        assert sp == h.spec_for(k)

