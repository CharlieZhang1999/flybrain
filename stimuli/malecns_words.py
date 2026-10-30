"""Stimulus vocabulary for MaleCNS v1.0, defined by annotated cell types.

Each word is a set of MaleCNS `type` names (body-annotations feather), driven
with Poisson input. Taste modalities come from FlyWire's cell_sub_class labels
for the same labellar GRN types (Schlegel et al. 2024 annotations):
  LB3c sugar, LB3b sugar/low salt, LB3a water, LB3d high salt/heavy metal,
  LB1a-d bitter, LB1e glutamate.
Olfactory words use ORN glomerulus types; thermo/hygro use TRN/HRN types
(VP2 hot, VP3 cold, VP4 dry, VP5 moist); hearing and wind use Johnston's
organ subgroups (JO-A/B sound, JO-C/E wind/gravity).
Visual input uses looming detectors (LPLC2, LC4) rather than photoreceptors:
photoreceptors are histaminergic (inhibitory), and a LIF network without
rebound excitation cannot turn their inhibition into a light response.
"""

WORDS = {
    'sugar':     dict(types=['LB3c'], modality='taste',
                      note='labellar sugar GRNs (Gr64f class)'),
    'bitter':    dict(types=['LB1a', 'LB1b', 'LB1c', 'LB1d'], modality='taste',
                      note='labellar bitter GRNs (Gr66a class)'),
    'water':     dict(types=['LB3a'], modality='taste',
                      note='labellar water GRNs (ppk28)'),
    'salt':      dict(types=['LB3d'], modality='taste',
                      note='high-salt / heavy-metal GRNs'),
    'umami':     dict(types=['LB1e'], modality='taste',
                      note='glutamate-sensing GRNs'),
    'vinegar':   dict(types=['ORN_DM1', 'ORN_DM4'], modality='smell',
                      note='acetate / apple-cider-vinegar glomeruli'),
    'fruit':     dict(types=['ORN_DM2', 'ORN_VA2'], modality='smell',
                      note='ester / ripe-fruit glomeruli'),
    'co2':       dict(types=['ORN_V'], modality='smell',
                      note='CO2 glomerulus V (stress odor)'),
    'mold':      dict(types=['ORN_DA2'], modality='smell',
                      note='geosmin glomerulus DA2 (danger)'),
    'pheromone': dict(types=['ORN_DA1', 'ORN_VA1v'], modality='smell',
                      note='cVA / fly-odor pheromone glomeruli'),
    'hot':       dict(types=['TRN_VP2'], modality='temperature',
                      note='antennal hot cells'),
    'cold':      dict(types=['TRN_VP3a', 'TRN_VP3b'], modality='temperature',
                      note='antennal cold cells'),
    'dry':       dict(types=['HRN_VP4'], modality='humidity',
                      note='dry-air hygrosensors'),
    'humid':     dict(types=['HRN_VP5'], modality='humidity',
                      note='moist-air hygrosensors'),
    'sound':     dict(prefix=['JO-A', 'JO-B'], modality='hearing',
                      note="Johnston's organ A/B (courtship song)"),
    'wind':      dict(prefix=['JO-C', 'JO-E'], modality='mechanosense',
                      note="Johnston's organ C/E (wind, gravity)"),
    'looming':   dict(types=['LPLC2', 'LC4'], modality='vision',
                      note='looming detectors (approaching object)'),
}

# free-text synonyms for the frontend / CLI
SYNONYMS = {
    'sweet': 'sugar', 'candy': 'sugar', 'honey': 'sugar', 'sucrose': 'sugar',
    'nectar': 'sugar', 'juice': 'sugar',
    'quinine': 'bitter', 'coffee': 'bitter', 'caffeine': 'bitter',
    'poison': 'bitter', 'toxin': 'bitter',
    'drink': 'water', 'thirst': 'water',
    'salty': 'salt', 'brine': 'salt',
    'glutamate': 'umami', 'msg': 'umami', 'savory': 'umami',
    'acetic': 'vinegar', 'cider': 'vinegar', 'sour': 'vinegar',
    'banana': 'fruit', 'apple': 'fruit', 'yeast': 'fruit', 'ester': 'fruit',
    'carbon': 'co2', 'breath': 'co2',
    'geosmin': 'mold', 'rot': 'mold', 'mould': 'mold',
    'cva': 'pheromone', 'mate': 'pheromone', 'female': 'pheromone',
    'heat': 'hot', 'warm': 'hot', 'fire': 'hot',
    'chill': 'cold', 'cool': 'cold', 'ice': 'cold',
    'desert': 'dry', 'moist': 'humid', 'wet': 'humid', 'rain': 'humid',
    'song': 'sound', 'music': 'sound', 'noise': 'sound', 'loud': 'sound',
    'breeze': 'wind', 'air': 'wind', 'gust': 'wind',
    'predator': 'looming', 'threat': 'looming', 'shadow': 'looming',
    'swatter': 'looming', 'danger': 'looming',
}


def resolve(word):
    w = word.strip().lower()
    return w if w in WORDS else SYNONYMS.get(w)


def select_ids(annot, word):
    """Return bodyIds (np.ndarray) for a word given the annotation table."""
    spec = WORDS[word]
    t = annot['type'].fillna('')
    m = t.isin(spec.get('types', []))
    for p in spec.get('prefix', []):
        m |= t.str.startswith(p)
    return annot.loc[m, 'bodyId'].to_numpy()
