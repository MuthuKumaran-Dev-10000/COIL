# Create updated enc.py and dec.py in the current working directory (not /mnt/data).
# Encoder will generate key and value tokens intelligently:
#  - prefer human-readable abbreviations (T for temperature, mdu for madurai)
#  - ensure tokens do not appear anywhere in the payload text (to avoid ambiguity)
#  - avoid collisions; append numeric suffixes if needed
#  - value tokens use lowercase like 'mdu', key tokens use uppercase like 'T'
#
# Then demonstrate with an example and show round-trip.
from pathlib import Path
import textwrap, json, re

enc_code = textwrap.dedent(r"""

import zlib, json, re
from collections import Counter

ESCAPE_CHAR = '\\\\'
PAIR_SEP = ','
RECORD_SEP = '|'

def _escape_value(v: str) -> str:
    return v.replace(ESCAPE_CHAR, ESCAPE_CHAR+ESCAPE_CHAR).replace(':', ESCAPE_CHAR+':').replace('|', ESCAPE_CHAR+'|').replace(',', ESCAPE_CHAR+',')

def _flatten_records(records):
    if isinstance(records, list):
        return records
    if isinstance(records, dict) and len(records)==1:
        first = next(iter(records.values()))
        if isinstance(first, list):
            return first
    raise ValueError('Unsupported sensordata format; expected list of dicts or a single-key dict -> list')

def _normalize_text(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def _candidate_key_tokens(longkey):
    # yield candidate uppercase tokens for a key
    sk = re.findall(r'[A-Za-z]+', longkey)
    if len(sk) > 1:
        # acronym of words
        yield ''.join([w[0] for w in sk]).upper()
    # prefer single letter if meaningful
    yield sk[0][0].upper() if sk else longkey[:1].upper()
    # first two letters
    yield (sk[0][:2].upper() if sk else longkey[:2].upper())
    # first three
    yield (sk[0][:3].upper() if sk else longkey[:3].upper())

def _candidate_value_tokens(val):
    v = str(val)
    words = re.findall(r'[A-Za-z0-9]+', v)
    # try acronym from multiple words (e.g., New York -> ny)
    if len(words) > 1:
        yield ''.join([w[0] for w in words]).lower()
    # compact by removing vowels (keep consonants)
    nv = re.sub(r'[aeiou]+','', ''.join(words).lower())
    if nv:
        yield nv[:3]
        yield nv[:4]
    # first 3 letters
    if words:
        yield words[0][:3].lower()
        yield words[0][:2].lower()
    # fallback numeric-like token
    yield re.sub(r'[^0-9]', '', v) or v[:3].lower()

def _pick_token(candidates, existing_tokens, payload_text):
    # pick first candidate that doesn't appear in payload_text and not in existing tokens
    for c in candidates:
        if not c:
            continue
        token = re.sub(r'[^A-Za-z0-9]', '', str(c))
        if token == '':
            continue
        # avoid tokens that appear as substrings in payload text
        if token.lower() in payload_text:
            continue
        if token in existing_tokens:
            continue
        return token
    # if none safe, return a numeric-suffixed token
    base = candidates[0] if candidates else 'x'
    base = re.sub(r'[^A-Za-z0-9]', '', str(base)) or 't'
    i = 1
    token = f\"{base}{i}\"
    while token in existing_tokens or token.lower() in payload_text:
        i += 1
        token = f\"{base}{i}\"
    return token

def _make_short_keys(all_keys, payload_text, preferred_map=None):
    short = {}
    used = set()
    preferred_map = preferred_map or {}
    for k in sorted(all_keys):
        if k in preferred_map:
            cand = [preferred_map[k]]
        else:
            cand = list(_candidate_key_tokens(k))
        token = _pick_token(cand, used, payload_text)
        # ensure uppercase for keys
        token = token.upper()
        short[k] = token
        used.add(token)
    return short

def _make_value_map(all_values, payload_text, min_freq=2, min_len=1):
    # Return mapping short->long for repeated values
    cnt = Counter(all_values)
    vmap = {}
    used = set()
    # sort values by frequency then length (desc freq, desc len)
    items = sorted(cnt.items(), key=lambda x: (-x[1], -len(str(x[0]))))
    idx = 1
    for val, freq in items:
        s = str(val)
        if freq < min_freq or len(s) < min_len:
            continue
        candidates = list(_candidate_value_tokens(s))
        token = _pick_token(candidates, used, payload_text)
        # ensure lowercase for vmap tokens
        token = token.lower()
        # avoid collision with keys or previous tokens; append number if needed
        if token in used:
            token = f\"{token}{idx}\"
            idx += 1
        used.add(token)
        vmap[token] = s
    return vmap

def encode(payload, preferred_map=None, value_min_freq=2):
    obj = dict(payload)
    if 'data' not in obj:
        return obj
    data = obj['data']
    if isinstance(data, dict) and 'sensordata' in data:
        records = _flatten_records(data['sensordata'])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(\"Unsupported 'data' structure for encoding. Provide {'sensordata': [...] } or a list of records.\")
    for r in records:
        if not isinstance(r, dict):
            raise ValueError('Each record must be a JSON object/dict')

    # collect keys and values
    all_keys = set()
    all_values = []
    for r in records:
        all_keys.update(r.keys())
        for v in r.values():
            if v is None:
                continue
            all_values.append(str(v))

    # build payload text for collision checks (lowercased)
    payload_text = json_text = json.dumps(payload, ensure_ascii=False).lower()

    short_map = _make_short_keys(all_keys, payload_text, preferred_map=preferred_map or {})

    # build value abbreviation map for repeated values
    vmap = _make_value_map(all_values, payload_text, min_freq=value_min_freq)
    vmap_long_to_short = {v: k for k, v in vmap.items()}

    # map entries and vmap entries
    map_entries = [f\"{short_map[k]}:{k}\" for k in sorted(short_map.keys())]
    vmap_entries = [f\"{s}:{v}\" for s, v in vmap.items()]

    # header
    ordered_shorts = [short_map[k] for k in sorted(short_map.keys())]
    header = f\"sensordata[{len(records)}]{{{','.join(ordered_shorts)}}}\"

    # build rows
    row_texts = []
    for r in records:
        parts = []
        for k in sorted(short_map.keys()):
            sk = short_map[k]
            v = r.get(k, '')
            if v is None:
                v = ''
            v = str(v)
            if v in vmap_long_to_short:
                v_out = vmap_long_to_short[v]
            else:
                v_out = _escape_value(v)
            parts.append(f\"{sk}:{v_out}\")
        row_texts.append(PAIR_SEP.join(parts))

    body_payload = RECORD_SEP.join([header] + row_texts)
    checksum = format(zlib.crc32(body_payload.encode('utf-8')) & 0xffffffff, '08x')

    meta_parts = [f\"checksum={checksum}\", f\"map={';'.join(map_entries)}\"]
    if vmap_entries:
        meta_parts.append(f\"vmap={';'.join(vmap_entries)}\")
    for pick in ('q', 'mdu'):
        if pick in payload:
            meta_parts.append(f\"{pick}={payload[pick]}\")

    meta_str = 'META&' + '&'.join(meta_parts)
    body_str = 'BODY|' + body_payload

    obj['data'] = {'meta': meta_str, 'body': body_str}
    return obj

if __name__ == '__main__':
    import sys, json, pathlib
    if len(sys.argv) < 2:
        print('Usage: python enc.py input.json')
        raise SystemExit(1)
    p = pathlib.Path(sys.argv[1])
    payload = json.loads(p.read_text(encoding='utf8'))
    print(json.dumps(encode(payload), indent=2, ensure_ascii=False))
""")

dec_code = textwrap.dedent(r"""
\"\"\"dec.py - COIL decoder (handles vmap value expansion)
\"\"\"
import zlib, re, json
ESCAPE_CHAR = '\\\\'
PAIR_SEP = ','
RECORD_SEP = '|'

def _unescape_value(v: str) -> str:
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

def decode(obj):
    new = dict(obj)
    if 'data' not in new or not isinstance(new['data'], dict):
        raise ValueError(\"Input JSON does not contain COIL-style 'data' dict with 'meta' and 'body'.\")

    meta = new['data'].get('meta','')
    body = new['data'].get('body','')

    if not meta.startswith('META&') or not body.startswith('BODY|'):
        raise ValueError('Not valid COIL META/BODY format')

    meta_body = meta[len('META&'):]
    meta_parts = meta_body.split('&')
    meta_kv = {}
    for part in meta_parts:
        if '=' in part:
            k,v = part.split('=',1)
            meta_kv[k] = v

    if 'map' not in meta_kv:
        raise ValueError('META missing map entry')
    map_raw = meta_kv['map']
    entries = map_raw.split(';')
    short_to_long = {}
    for e in entries:
        if ':' in e:
            sk, lk = e.split(':',1)
            short_to_long[sk] = lk

    vmap = {}
    if 'vmap' in meta_kv and meta_kv['vmap']:
        for e in meta_kv['vmap'].split(';'):
            if ':' in e:
                sk, lv = e.split(':',1)
                vmap[sk] = lv

    expected_checksum = meta_kv.get('checksum')
    body_payload = body[len('BODY|'):]
    if expected_checksum:
        calc = format(zlib.crc32(body_payload.encode('utf-8')) & 0xffffffff, '08x')
        if calc != expected_checksum:
            raise ValueError(f'Checksum mismatch: expected {expected_checksum} got {calc}')

    parts = body_payload.split(RECORD_SEP)
    if not parts:
        return new

    header = parts[0]
    m = re.match(r'sensordata\\[(\\d+)\\]\\{(.+)\\}', header)
    if not m:
        ordered_shorts = list(short_to_long.keys())
        data_rows = parts
    else:
        order_list = m.group(2).split(',')
        ordered_shorts = order_list
        data_rows = parts[1:]

    records = []
    for r in data_rows:
        if r.strip() == '':
            continue
        kvs = r.split(PAIR_SEP)
        rec = {}
        for kv in kvs:
            if ':' not in kv:
                continue
            sk, vv = kv.split(':',1)
            if vv in vmap:
                val = vmap[vv]
            else:
                val = _unescape_value(vv)
            longk = short_to_long.get(sk, sk)
            rec[longk] = val
        records.append(rec)

    new['data'] = {'sensordata': records}
    return new

if __name__ == '__main__':
    import sys, json, pathlib
    if len(sys.argv) < 2:
        print('Usage: python dec.py coil_wrapped.json')
        raise SystemExit(1)
    p = pathlib.Path(sys.argv[1])
    payload = json.loads(p.read_text(encoding='utf8'))
    print(json.dumps(decode(payload), indent=2, ensure_ascii=False))
""")

# write files to current directory
Path('enc.py').write_text(enc_code, encoding='utf8')
Path('dec.py').write_text(dec_code, encoding='utf8')

print("Wrote ./enc.py and ./dec.py\n")

# Demonstrate with example payload and round-trip
example = {
    "device": "sensor-xyz",
    "q": 2025,
    "mdu": "Madurai",
    "data": {
        "sensordata": [
            {"temperature":"49","Time":"3-8-2025","place":"madurai"},
            {"temperature":"35","Time":"5-9-2025","place":"chennai"},
            {"temperature":"35","Time":"2-3-2025","place":"madurai"}
        ]
    }
}

# import modules dynamically and run round-trip
import importlib.util, sys
spec = importlib.util.spec_from_file_location("enc_module", "enc.py")
enc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc_module)

spec2 = importlib.util.spec_from_file_location("dec_module", "dec.py")
dec_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(dec_module)

encoded = enc_module.encode(example, preferred_map={'temperature':'T', 'place':'P'})
print("Encoded payload:\n")
print(json.dumps(encoded, indent=2, ensure_ascii=False))

decoded = dec_module.decode(encoded)
print("\nDecoded payload:\n")
print(json.dumps(decoded, indent=2, ensure_ascii=False))

# Save outputs to current directory
Path('example_original.json').write_text(json.dumps(example, indent=2), encoding='utf8')
Path('example_encoded.json').write_text(json.dumps(encoded, indent=2), encoding='utf8')
Path('example_decoded.json').write_text(json.dumps(decoded, indent=2), encoding='utf8')

print("\\nSaved example files in current directory.")
