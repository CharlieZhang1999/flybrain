#!/usr/bin/env python
"""C7b step 1: extract per-T-bar transmitter probabilities for antennal-lobe
local and projection neurons (class ALLN / ALPN) from the 2.65 GB MaleCNS
tbar-neurotransmitters table. Output: data/malecns/tbar_nt_AL.parquet
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data' / 'malecns'


def main():
    meta = pd.read_parquet(D / 'malecns_traced_meta.parquet')
    ids = pa.array(meta.bodyId[meta['class'].isin(['ALLN', 'ALPN'])].astype('int64').to_numpy())
    cols = ['body', 'conf'] + [c for c in ipc.open_file(D / 'tbar-neurotransmitters-male-cns-v1.0.feather').schema.names
                               if c.startswith('nt_')]
    r = ipc.open_file(D / 'tbar-neurotransmitters-male-cns-v1.0.feather')
    parts = []
    for k in range(r.num_record_batches):
        b = r.get_batch(k).select(cols)
        parts.append(b.filter(pc.is_in(b['body'], value_set=ids)))
    t = pa.Table.from_batches(parts)
    pq.write_table(t, D / 'tbar_nt_AL.parquet')
    print(f'{len(ids)} AL neurons, {t.num_rows:,} T-bars saved')


if __name__ == '__main__':
    main()
