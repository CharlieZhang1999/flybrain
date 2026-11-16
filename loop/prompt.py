"""Fly -> LLM prompt template (Week 3 Day 2). Decoded words become first-person
phrases by modality (stimuli/malecns_words.py); the LLM must answer with one of
five actions."""
ACTIONS = ['turn left', 'turn right', 'go forward', 'stop', 'eat']

PHRASE = {
    'sugar': 'I taste sugar.', 'bitter': 'I taste something bitter.', 'water': 'I taste water.',
    'salt': 'I taste salt.', 'umami': 'I taste something savory.',
    'vinegar': 'I smell vinegar.', 'fruit': 'I smell ripe fruit.', 'co2': 'I smell CO2.',
    'mold': 'I smell mould.', 'pheromone': 'I smell another fly.',
    'hot': 'It feels hot.', 'cold': 'It feels cold.', 'dry': 'The air feels dry.',
    'humid': 'The air feels humid.', 'sound': 'I hear a sound.', 'wind': 'I feel wind.',
    'looming': 'Something is looming toward me.',
}

SYSTEM = ('You are a fruit fly. You are told what you sense, in the order you sensed it. '
          'Choose exactly one action: ' + ', '.join(ACTIONS) + '.')


def build(decoded):
    """decoded: [(word, p)] per window. 'none' and 'unsure' windows are skipped."""
    lines = [PHRASE[w] for w, _ in decoded if w in PHRASE]
    if not lines:
        lines = ['I sense nothing.']
    return ' '.join(lines) + ' What should I do?'
