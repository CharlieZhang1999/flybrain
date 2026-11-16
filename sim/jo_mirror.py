"""Mirror calibration for right-side Johnston's organ (JO) afferents (optional, off by default).

Why: in MaleCNS v1.0 the right JO afferents are much less complete than the left
(all 672 JO bodies are Traced, 348 L / 324 R, but right bodies make 67.6k output
synapses vs 197.4k on the left; 24 of 32 right JO-A and 95 of 102 JO-unclear
bodies are on the right and most of them have no output synapses). In FlyWire the
two sides are about equal (43k L / 48k R output synapses in Connectivity_783), and
JO afferents project ipsilaterally into the AMMC in a bilaterally symmetric zone
map (Kamikouchi et al. 2006; Matsuo et al. 2014). So the right deficit is treated
as a reconstruction artifact. See progress/week3/right_jo_fix.md.

What it does (only the COLUMNS = outputs of right JO neurons change):
1. For each left JO subtype s (MaleCNS `type`, e.g. JO-A1), take the mean output
   per left-s neuron onto each partner type T on each side X:
       m[s, T, X] = sum_{j in T_X} W[j, left s] / n_left_s
2. Mirror: give that total to the homologs of T on the opposite side, split evenly
   across them (exact one-to-one when a type has one neuron per side). Partners on
   the midline (no side) map to themselves. Untyped partners, or types with no
   neuron on the opposite side, are dropped.
3. Each right JO neuron gets the template of its own subtype. Right neurons whose
   subtype has no left counterpart, or that are typed "unclear" (JO-A-unclear,
   JO-B-unclear, JO-unclear), get the template with the best cosine overlap with
   their existing (fragment) outputs, among the templates of the same group
   (JO-A-unclear -> JO-A*, JO-B-unclear -> JO-B*, JO-unclear -> any JO). With no
   outputs to compare, JO-A/B-unclear get the group mean (generic JO-A / JO-B);
   JO-unclear neurons with no outputs are left unchanged (their zone is unknown).
The number of right neurons is kept as in the data.
"""
import numpy as np
import pandas as pd
from scipy import sparse


def _side(meta):
    """'L'/'R' from the instance suffix, else somaSide, else '' (midline/unknown)."""
    inst = meta['instance'].astype(str)
    s = np.where(inst.str.endswith('_L'), 'L', np.where(inst.str.endswith('_R'), 'R', ''))
    soma = meta['somaSide'].astype(str).to_numpy() if 'somaSide' in meta else np.full(len(meta), '')
    s = np.where((s == '') & np.isin(soma, ['L', 'R']), soma, s)
    return s


def _group(t):
    """'A'/'B'/... for JO-A1 etc.; 'any' for JO-unclear."""
    if t == 'JO-unclear':
        return 'any'
    return t[3:].split('-')[0].rstrip('0123456789').split('_')[0].rstrip('0123456789') or 'any'


def mirror_jo(W, meta, mode='replace', source='L'):
    """W: post x pre csc (volts), meta: model metadata (rows/cols of W).
    mode: 'replace' (right JO outputs := mirrored template) or 'max' (elementwise
    max of existing and template, keeps any right-specific contacts).
    source: side used as template ('L' fixes the right side).
    Returns (W_new csc, info dict)."""
    W = sparse.csc_matrix(W)
    n = W.shape[0]
    typ = meta['type'].astype(str).to_numpy()
    side = _side(meta)
    tgt_side = 'R' if source == 'L' else 'L'
    opp = {'L': 'R', 'R': 'L', '': ''}
    isjo = pd.Series(typ).str.startswith('JO-').to_numpy()
    src = np.flatnonzero(isjo & (side == source))
    dst = np.flatnonzero(isjo & (side == tgt_side))

    # homolog index: (type, side) -> neuron indices
    key = pd.Series(np.arange(n)).groupby([typ, side]).apply(np.array).to_dict()
    nan_types = {'nan', 'None', ''}

    def mirrored_template(cols):
        """Mean output column of `cols`, mirrored onto contralateral homologs."""
        sub = W[:, cols]
        v = np.asarray(sub.sum(axis=1)).ravel() / len(cols)
        rows = np.flatnonzero(v)
        df = pd.DataFrame(dict(T=typ[rows], X=side[rows], w=v[rows]))
        df = df[~df['T'].isin(nan_types)]
        tot = df.groupby(['T', 'X']).w.sum()
        out = np.zeros(n); dropped = 0.0
        for (T, X), w in tot.items():
            h = key.get((T, opp[X]))
            if h is None or len(h) == 0:
                dropped += w; continue
            out[h] += w / len(h)
        kept = out.sum()
        return out, dict(total=float(v.sum()), untyped=float(v.sum() - df.w.sum()),
                         no_homolog=float(dropped), kept=float(kept))

    templates, tinfo = {}, {}
    for s in sorted(set(typ[src])):
        cols = src[typ[src] == s]
        if 'unclear' in s:      # unclear left neurons are not a defined subtype
            continue
        templates[s], tinfo[s] = mirrored_template(cols)
        tinfo[s]['n_src'] = int(len(cols))
    generic = {}
    for g in ['A', 'B']:
        cols = src[[(_group(t) == g and 'unclear' not in t) for t in typ[src]]]
        if len(cols):
            generic[g], tinfo[f'JO-{g} (generic)'] = mirrored_template(cols)

    rows_l, cols_l, vals_l, before, after = [], [], [], [], []
    assign = []
    for j in dst:
        t = typ[j]; old = W[:, j].toarray().ravel()
        g = _group(t)
        cands = [s for s in templates if g == 'any' or _group(s) == g]
        if t in templates:
            new, a = templates[t], (j, t, t, 'same type', np.nan)
        else:
            new = None
            if old.any() and cands:
                T = np.array([templates[s] for s in cands])
                cos = T @ old / (np.linalg.norm(T, axis=1) * np.linalg.norm(old) + 1e-30)
                b = int(np.argmax(cos))
                if cos[b] > 0:
                    new, a = templates[cands[b]], (j, t, cands[b], 'best overlap', float(cos[b]))
            if new is None and g in generic:
                new, a = generic[g], (j, t, f'JO-{g} (generic)', 'group mean', np.nan)
            if new is None:
                new, a = old, (j, t, '', 'unchanged (no outputs, zone unknown)', np.nan)
        if mode == 'max':
            new = np.maximum(new, old)
        r = np.flatnonzero(new)
        rows_l.append(r); cols_l.append(np.full(len(r), j)); vals_l.append(new[r])
        before.append(old.sum()); after.append(new.sum()); assign.append(a)

    # rebuild: every other column is untouched
    keep = np.ones(n, bool); keep[dst] = False
    Wk = (W @ sparse.diags(keep.astype(W.dtype))).tocsc()
    Wn = sparse.csc_matrix((np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
                           shape=W.shape, dtype=W.dtype)
    W_new = (Wk + Wn).tocsc(); W_new.eliminate_zeros()
    A = pd.DataFrame(assign, columns=['idx', 'type', 'template', 'rule', 'cosine'])
    A['out_before'] = before; A['out_after'] = after
    info = dict(mode=mode, source=source, n_src=int(len(src)), n_dst=int(len(dst)),
                templates=tinfo, assign=A,
                out_src_total=float(W[:, src].sum()), out_dst_before=float(np.sum(before)),
                out_dst_after=float(np.sum(after)))
    return W_new, info
