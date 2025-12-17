# enc.py - COIL encoder (compact mode ON by default)
# - Default behavior: compact mode -> write META with ORDER=key1,key2,... and body rows as positional CSV.
# - Optional vmap still applied only if it reduces token count.
# - Decoder supports both compact positional format (ORDER + positional rows)
#   and legacy format (map + key:value rows).
# - No checksum. Uses tiktoken for accurate token decisions if available.

import json
import re
from collections import Counter

ESCAPE_CHAR = '\\'
PAIR_SEP = ','
RECORD_SEP = '|'

# Try to use tiktoken
try:
    import tiktoken
    _TIKTOKEN = True
    _MODEL = "gpt-4o-mini"
    _ENC = tiktoken.encoding_for_model(_MODEL)
except Exception:
    _TIKTOKEN = False
    _ENC = None

def _token_count(text: str) -> int:
    if _TIKTOKEN and _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, (len(text) + 3) // 4)

def _escape_val(v: str) -> str:
    # escape colon, pipe, comma and backslash so legacy format remains safe;
    # for compact positional rows we still escape commas and pipes.
    return (v.replace(ESCAPE_CHAR, ESCAPE_CHAR+ESCAPE_CHAR)
             .replace(':', ESCAPE_CHAR+':')
             .replace('|', ESCAPE_CHAR+'|')
             .replace(',', ESCAPE_CHAR+','))

def _unescape_val(v: str) -> str:
    out = []
    i = 0
    while i < len(v):
        if v[i] == ESCAPE_CHAR and i+1 < len(v):
            out.append(v[i+1])
            i += 2
        else:
            out.append(v[i])
            i += 1
    return ''.join(out)

def _flatten_records(records):
    if isinstance(records, list):
        return records
    if isinstance(records, dict) and len(records) == 1:
        first = next(iter(records.values()))
        if isinstance(first, list):
            return first
    raise ValueError("Unsupported sensordata format; expected list of dicts or a single-key dict -> list")

def _word_collision(token: str, payload_text: str) -> bool:
    if not token:
        return False
    try:
        return re.search(r'\b' + re.escape(token.lower()) + r'\b', payload_text) is not None
    except re.error:
        return token.lower() in payload_text

def _build_positional_body(records, key_order, long_to_token):
    """Build compact positional body: header row is keys (for human-readability) but rows are positional values."""
    header = f"sensordata[{len(records)}]{{{','.join(key_order)}}}"
    rows = []
    for r in records:
        vals = []
        for k in key_order:
            v = r.get(k, '')
            if v is None:
                v = ''
            vs = str(v)
            if vs in long_to_token:
                outv = long_to_token[vs]
            else:
                outv = _escape_val(vs)
            vals.append(outv)
        rows.append(PAIR_SEP.join(vals))  # single-line row
    return RECORD_SEP.join([header] + rows)

def _propose_value_tokens(records, payload_text, min_freq=2):
    all_values = []
    for r in records:
        for v in r.values():
            if v is None:
                continue
            all_values.append(str(v))
    cnt = Counter(all_values)
    candidates = [v for v, f in cnt.items() if f >= min_freq]
    candidates.sort(key=lambda v: (cnt[v] * len(str(v))), reverse=True)
    proposals = {}
    i = 1
    used = set()
    for v in candidates:
        tok = f"V{i}"
        if _word_collision(tok, payload_text) or tok in used:
            n = 1
            cand = f"{tok}{n}"
            while _word_collision(cand, payload_text) or cand in used:
                n += 1
                cand = f"{tok}{n}"
            tok = cand
        proposals[v] = tok
        used.add(tok)
        i += 1
    return proposals, cnt

def encode(payload, value_min_freq=2, compact=True):
    """
    payload: dict like { ..., "data": {"sensordata": [ {...}, ... ] } }
    value_min_freq: min repetition to consider value mapping
    compact: if True (default) use positional compact format (ORDER + positional rows)
    """
    obj = dict(payload)
    if 'data' not in obj:
        return obj
    data = obj['data']
    records = _flatten_records(data['sensordata']) if isinstance(data, dict) and 'sensordata' in data else _flatten_records(data)
    for r in records:
        if not isinstance(r, dict):
            raise ValueError("Each record must be a JSON object/dict")

    # decide keys / order
    key_order = sorted({k for r in records for k in r.keys()})

    payload_text = json.dumps(payload, ensure_ascii=False).lower()

    # baseline tokens (no value mapping), two variants: compact positional vs legacy key:value
    if compact:
        base_body = _build_positional_body(records, key_order, {})
        base_meta = 'META&' + f"ORDER={','.join(key_order)}"
    else:
        # legacy key:value header & rows
        header = f"sensordata[{len(records)}]{{{','.join(key_order)}}}"
        rows = []
        for r in records:
            parts = []
            for k in key_order:
                v = r.get(k, '')
                if v is None:
                    v = ''
                parts.append(f"{k}:{_escape_val(str(v))}")
            rows.append(PAIR_SEP.join(parts))
        base_body = RECORD_SEP.join([header] + rows)
        base_meta = 'META&' + f"map={';'.join([f'{k}:{k}' for k in key_order])}"

    base_total_tokens = _token_count(base_meta + '|' + base_body)

    # propose vmap candidates
    proposed_vals, cnt = _propose_value_tokens(records, payload_text, min_freq=value_min_freq)
    if not proposed_vals:
        final_vmap = {}
        final_body = base_body
        final_meta = base_meta
    else:
        accepted = {}
        accepted_vmap = {}
        current_tokens = base_total_tokens
        for val in proposed_vals:
            tok = proposed_vals[val]
            if tok in accepted_vmap or _word_collision(tok, payload_text):
                n = 1
                cand = f"{tok}{n}"
                while cand in accepted_vmap or _word_collision(cand, payload_text):
                    n += 1
                    cand = f"{tok}{n}"
                tok = cand
            hypot_long_to_token = dict(accepted)
            hypot_long_to_token[val] = tok
            if compact:
                hypot_body = _build_positional_body(records, key_order, hypot_long_to_token)
                meta_parts = [f"ORDER={','.join(key_order)}"]
            else:
                # legacy map + key:value rows
                header = f"sensordata[{len(records)}]{{{','.join(key_order)}}}"
                rows = []
                for r in records:
                    parts = []
                    for k in key_order:
                        v = r.get(k, '')
                        if v is None:
                            v = ''
                        vs = str(v)
                        if vs in hypot_long_to_token:
                            outv = hypot_long_to_token[vs]
                        else:
                            outv = _escape_val(vs)
                        parts.append(f"{k}:{outv}")
                    rows.append(PAIR_SEP.join(parts))
                hypot_body = RECORD_SEP.join([header] + rows)
                meta_parts = [f"map={';'.join([f'{k}:{k}' for k in key_order])}"]
            # include vmap entries in meta for token calc
            vm_entries = [f"{t}:{v}" for v, t in hypot_long_to_token.items()]
            if vm_entries:
                meta_parts.append(f"vmap={';'.join(vm_entries)}")
            hypot_meta = 'META&' + '&'.join(meta_parts)
            hypot_total_tokens = _token_count(hypot_meta + '|' + hypot_body)
            gain = current_tokens - hypot_total_tokens
            if gain > 0:
                accepted[val] = tok
                accepted_vmap[tok] = val
                current_tokens = hypot_total_tokens
            # else skip
        final_vmap = {tok: val for tok, val in accepted_vmap.items()}
        if compact:
            final_body = _build_positional_body(records, key_order, accepted)
            final_meta = 'META&' + f"ORDER={','.join(key_order)}"
        else:
            # build legacy body with accepted mapping
            header = f"sensordata[{len(records)}]{{{','.join(key_order)}}}"
            rows = []
            for r in records:
                parts = []
                for k in key_order:
                    v = r.get(k, '')
                    if v is None:
                        v = ''
                    vs = str(v)
                    if vs in accepted:
                        outv = accepted[vs]
                    else:
                        outv = _escape_val(vs)
                    parts.append(f"{k}:{outv}")
                rows.append(PAIR_SEP.join(parts))
            final_body = RECORD_SEP.join([header] + rows)
            final_meta = 'META&' + f"map={';'.join([f'{k}:{k}' for k in key_order])}"

        # add vmap entries to meta if any
        vm_entries = [f"{tok}:{val}" for tok, val in final_vmap.items()]
        meta_parts = [final_meta[len('META&'):]]  # existing meta body (ORDER=... or map=...)
        if vm_entries:
            meta_parts.append(f"vmap={';'.join(vm_entries)}")
        # also include q/mdu passthrough if present
    # assemble final meta/body and attach
    meta_parts = [final_meta[len('META&'):]] if 'final_meta' in locals() else [base_meta[len('META&'):]]
    if final_vmap:
        meta_parts.append(f"vmap={';'.join([f'{k}:{v}' for k, v in final_vmap.items()])}")
    # top-level extras
    for pick in ('q', 'mdu'):
        if pick in payload:
            meta_parts.append(f"{pick}={payload[pick]}")
    meta_str = 'META&' + '&'.join(meta_parts)
    obj['data'] = {'meta': meta_str, 'body': 'BODY|' + (final_body if 'final_body' in locals() else base_body)}
    return obj
