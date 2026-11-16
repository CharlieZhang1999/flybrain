"""LLM client for the fly loop (Week 3 Day 2).

provider: 'stub' (offline rule-based policy, default when no key is set),
'claude-cli' (Claude through the local Claude Code CLI, `claude -p`, using your
Claude login; no API key; model CLAUDE_CLI_MODEL if set),
'openai' (OPENAI_API_KEY, model OPENAI_MODEL) or 'anthropic' (ANTHROPIC_API_KEY,
model ANTHROPIC_MODEL, default claude-sonnet-5-5). Keys are read from the
environment or flybrain/.env. The reply is forced to one of prompt.ACTIONS via a
tool call with an enum; the raw reply is always returned for the log, and
anything outside the list maps to 'stop'.
"""
import json
import os
from pathlib import Path
from loop.prompt import ACTIONS, SYSTEM

ROOT = Path(__file__).resolve().parents[1]
TOOL_DESC = 'Report the single action the fly takes.'


def _env():
    f = ROOT / '.env'
    if f.exists():
        for line in f.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def stub(prompt):
    """Offline policy, in priority order (danger first, then food, then smells)."""
    p = prompt.lower()
    if 'looming' in p:
        return 'turn left'
    if 'bitter' in p or 'mould' in p or 'co2' in p or 'hot' in p:
        return 'stop' if 'sugar' not in p else 'turn right'
    if 'sugar' in p or 'savory' in p or 'water' in p:
        return 'eat'
    if 'vinegar' in p or 'ripe fruit' in p or 'another fly' in p:
        return 'go forward'
    return 'stop'


def ask(prompt, provider=None, temperature=0.0):
    """Returns dict(provider, model, action, raw)."""
    _env()
    if provider is None:
        provider = ('openai' if os.environ.get('OPENAI_API_KEY') else
                    'anthropic' if os.environ.get('ANTHROPIC_API_KEY') else 'stub')
    if provider == 'stub':
        a = stub(prompt)
        return dict(provider='stub', model='rules-v1', action=a, raw=a)
    if provider == 'openai':
        from openai import OpenAI
        model = os.environ.get('OPENAI_MODEL', 'gpt-4o')
        r = OpenAI().chat.completions.create(
            model=model, temperature=temperature,
            messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
            tools=[{'type': 'function', 'function': {
                'name': 'act', 'description': TOOL_DESC,
                'parameters': {'type': 'object', 'required': ['action'],
                               'properties': {'action': {'type': 'string', 'enum': ACTIONS}}}}}],
            tool_choice={'type': 'function', 'function': {'name': 'act'}})
        raw = r.choices[0].message.tool_calls[0].function.arguments
        action = json.loads(raw).get('action')
    elif provider == 'claude-cli':
        import subprocess
        model = os.environ.get('CLAUDE_CLI_MODEL', '')
        cmd = ['claude', '-p', prompt, '--output-format', 'json', '--system-prompt',
               SYSTEM + ' Reply with the action only, exactly as written, nothing else.']
        if model:
            cmd += ['--model', model]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=True).stdout
        raw = json.loads(out).get('result', '').strip()
        low = raw.lower().strip(' .\n"\'')
        action = low if low in ACTIONS else next((a for a in ACTIONS if a in low), None)
        model = model or 'claude-cli default'
    elif provider == 'anthropic':
        import anthropic
        model = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-5-5')
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=200, temperature=temperature, system=SYSTEM,
            messages=[{'role': 'user', 'content': prompt}],
            tools=[{'name': 'act', 'description': TOOL_DESC,
                    'input_schema': {'type': 'object', 'required': ['action'],
                                     'properties': {'action': {'type': 'string', 'enum': ACTIONS}}}}],
            tool_choice={'type': 'tool', 'name': 'act'})
        use = next(b for b in r.content if b.type == 'tool_use')
        raw = json.dumps(use.input); action = use.input.get('action')
    else:
        raise ValueError(provider)
    return dict(provider=provider, model=model, raw=raw,
                action=action if action in ACTIONS else 'stop')
