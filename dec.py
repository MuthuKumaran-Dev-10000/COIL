# dec.py - COIL decoder supporting compact ORDER positional format and legacy map:key:value format
import re
import json

ESCAPE_CHAR = '\\'
PAIR_SEP = ','
RECORD_SEP = '|'

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

def decode(obj):
    new = dict(obj)
    if 'data' not in new or not isinstance(new['data'], dict):
        raise ValueError("Input JSON must contain 'data' dict with 'meta' and 'body'.")

    meta = new['data'].get('meta', '')
    body = new['data'].get('body', '')

    if not meta.startswith('META&') or not body.startswith('BODY|'):
        raise ValueError("Not valid COIL META/BODY format")

    meta_body = meta[len('META&'):]
    meta_parts = meta_body.split('&') if meta_body else []
    meta_kv = {}
    for p in meta_parts:
        if '=' in p:
            k, v = p.split('=', 1)
            meta_kv[k] = v

    # Parse vmap if present
    vmap = {}
    if 'vmap' in meta_kv and meta_kv['vmap']:
        for e in meta_kv['vmap'].split(';'):
            if ':' in e:
                tok, val = e.split(':', 1)
                vmap[tok] = val

    body_payload = body[len('BODY|'):]
    parts = body_payload.split(RECORD_SEP)
    if not parts:
        return new

    header = parts[0]
    m = re.match(r'sensordata\[(\d+)\]\{(.+)\}', header)
    if not m:
        raise ValueError("Invalid BODY header")
    key_list = m.group(2).split(',')

    # Two possible row formats:
    # 1) legacy key:value pairs per row (contains ':' separators)
    # 2) compact positional rows: just comma-separated values aligned to key_list
    data_rows = parts[1:]
    records = []
    for row in data_rows:
        if row.strip() == '':
            continue
        # detect if row contains any ':' -> legacy key:value style
        if ':' in row:
            kvs = row.split(PAIR_SEP)
            rec = {}
            for kv in kvs:
                if ':' not in kv:
                    continue
                sk, vv = kv.split(':', 1)
                # if vv is in vmap, expand; else unescape
                if vv in vmap:
                    val = vmap[vv]
                else:
                    val = _unescape_val(vv)
                # map short header token back to original key name; if short==long (common), keep as-is
                # For legacy map, meta must contain 'map' mapping; if not, assume sk==sk
                # We'll reconstruct keys by checking meta 'map' if available
                rec[sk] = val
            # Attempt to remap header short tokens to original using meta 'map' if present
            # If meta contains 'map', use it
            if 'map' in meta_kv:
                short_to_long = {}
                for e in meta_kv['map'].split(';'):
                    if ':' in e:
                        s,l = e.split(':',1)
                        short_to_long[s] = l
                # remap keys
                rec2 = {}
                for sk, val in rec.items():
                    longk = short_to_long.get(sk, sk)
                    rec2[longk] = val
                records.append(rec2)
            else:
                records.append(rec)
        else:
            # compact positional row
            parts_vals = row.split(PAIR_SEP)
            rec = {}
            for i, k in enumerate(key_list):
                vraw = parts_vals[i] if i < len(parts_vals) else ''
                if vraw in vmap:
                    val = vmap[vraw]
                else:
                    val = _unescape_val(vraw)
                rec[k] = val
            records.append(rec)

    new['data'] = {'sensordata': records}
    return new
