# main.py - demo driver using new compact-first enc/dec
import json
from pathlib import Path
from enc import encode
from dec import decode
import math

try:
    import tiktoken
    TIK = True
    MODEL = "gpt-4o-mini"
    try:
        ENC = tiktoken.encoding_for_model(MODEL)
    except Exception:
        ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    TIK = False
    ENC = None
    MODEL = None

def token_count(text: str) -> int:
    if TIK and ENC is not None:
        try:
            return len(ENC.encode(text))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4))

def save_json(obj, path):
    p = Path(path)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')
    return p

def file_stats(p):
    txt = p.read_text(encoding='utf8')
    return {"path": str(p), "bytes": p.stat().st_size, "chars": len(txt), "tokens": token_count(txt), "text": txt}

def data_text(obj):
    if isinstance(obj, dict) and "data" in obj:
        return json.dumps(obj["data"], separators=(",", ":"), ensure_ascii=False)
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def demo():
    payload = {
    "device": "sensor-xyz",
    "area": "Tamilnadu",
    "data": {
        "sensordata": [
            {"temperature":"49","Time":"3-8-2025","place":"Madurai"},
            {"temperature":"35","Time":"5-9-2025","place":"Chennai"},
            {"temperature":"35","Time":"2-3-2025","place":"Madurai"},

            {"temperature":"42","Time":"11-1-2024","place":"Coimbatore"},
            {"temperature":"39","Time":"14-2-2024","place":"Trichy"},
            {"temperature":"51","Time":"9-3-2024","place":"Salem"},
            {"temperature":"47","Time":"28-4-2025","place":"Erode"},
            {"temperature":"33","Time":"15-5-2025","place":"Tirunelveli"},
            {"temperature":"55","Time":"7-6-2023","place":"Madurai"},
            {"temperature":"29","Time":"21-7-2023","place":"Chennai"},

            {"temperature":"38","Time":"10-8-2023","place":"Coimbatore"},
            {"temperature":"41","Time":"3-9-2023","place":"Trichy"},
            {"temperature":"36","Time":"19-10-2024","place":"Salem"},
            {"temperature":"44","Time":"26-11-2026","place":"Erode"},
            {"temperature":"32","Time":"5-12-2026","place":"Tirunelveli"},
            {"temperature":"28","Time":"6-1-2024","place":"Madurai"},
            {"temperature":"53","Time":"17-2-2025","place":"Chennai"},
            {"temperature":"37","Time":"25-3-2025","place":"Coimbatore"},
            {"temperature":"54","Time":"2-4-2025","place":"Trichy"},
            {"temperature":"46","Time":"14-5-2023","place":"Salem"},

            {"temperature":"33","Time":"18-6-2023","place":"Erode"},
            {"temperature":"45","Time":"9-7-2023","place":"Tirunelveli"},
            {"temperature":"31","Time":"27-8-2024","place":"Madurai"},
            {"temperature":"52","Time":"15-9-2024","place":"Chennai"},
            {"temperature":"40","Time":"30-10-2025","place":"Coimbatore"},
            {"temperature":"34","Time":"4-11-2025","place":"Trichy"},
            {"temperature":"48","Time":"12-12-2025","place":"Salem"},
            {"temperature":"50","Time":"19-1-2026","place":"Erode"},
            {"temperature":"43","Time":"23-2-2026","place":"Tirunelveli"},
            {"temperature":"29","Time":"3-3-2023","place":"Madurai"},

            # {"temperature":"47","Time":"7-4-2023","place":"Chennai"},
            # {"temperature":"35","Time":"22-5-2024","place":"Coimbatore"},
            # {"temperature":"32","Time":"14-6-2024","place":"Trichy"},
            # {"temperature":"37","Time":"29-7-2026","place":"Salem"},
            # {"temperature":"49","Time":"13-8-2026","place":"Erode"},
            # {"temperature":"44","Time":"2-9-2025","place":"Tirunelveli"},
            # {"temperature":"36","Time":"11-10-2025","place":"Madurai"},
            # {"temperature":"42","Time":"8-11-2023","place":"Chennai"},
            # {"temperature":"38","Time":"1-12-2023","place":"Coimbatore"},
            # {"temperature":"53","Time":"16-1-2024","place":"Trichy"},

            # {"temperature":"51","Time":"28-2-2024","place":"Salem"},
            # {"temperature":"35","Time":"9-3-2026","place":"Erode"},
            # {"temperature":"33","Time":"20-4-2026","place":"Tirunelveli"},
            # {"temperature":"46","Time":"6-5-2023","place":"Madurai"},
            # {"temperature":"41","Time":"18-6-2023","place":"Chennai"},
            # {"temperature":"29","Time":"25-7-2023","place":"Coimbatore"},
            # {"temperature":"45","Time":"7-8-2025","place":"Trichy"},
            # {"temperature":"30","Time":"18-9-2025","place":"Salem"},
            # {"temperature":"52","Time":"3-10-2023","place":"Erode"},
            # {"temperature":"39","Time":"27-11-2023","place":"Tirunelveli"},

            # {"temperature":"50","Time":"11-12-2026","place":"Madurai"},
            # {"temperature":"48","Time":"22-1-2026","place":"Chennai"},
            # {"temperature":"37","Time":"6-2-2026","place":"Coimbatore"},
            # {"temperature":"31","Time":"13-3-2024","place":"Trichy"},
            # {"temperature":"33","Time":"29-4-2024","place":"Salem"},
            # {"temperature":"55","Time":"15-5-2024","place":"Erode"},
            # {"temperature":"43","Time":"1-6-2024","place":"Tirunelveli"},
            # {"temperature":"32","Time":"19-7-2024","place":"Madurai"},
            # {"temperature":"46","Time":"4-8-2023","place":"Chennai"},
            # {"temperature":"35","Time":"12-9-2023","place":"Coimbatore"},

            # {"temperature":"30","Time":"20-10-2025","place":"Trichy"},
            # {"temperature":"38","Time":"28-11-2025","place":"Salem"},
            # {"temperature":"42","Time":"7-12-2025","place":"Erode"},
            # {"temperature":"39","Time":"14-1-2026","place":"Tirunelveli"},
            # {"temperature":"50","Time":"25-2-2026","place":"Madurai"},
            # {"temperature":"48","Time":"3-3-2024","place":"Chennai"},
            # {"temperature":"28","Time":"15-4-2024","place":"Coimbatore"},
            # {"temperature":"47","Time":"27-5-2024","place":"Trichy"},
            # {"temperature":"34","Time":"6-6-2024","place":"Salem"},
            # {"temperature":"43","Time":"18-7-2024","place":"Erode"},

            # {"temperature":"36","Time":"29-8-2024","place":"Tirunelveli"},
            # {"temperature":"52","Time":"9-9-2024","place":"Madurai"},
            # {"temperature":"33","Time":"21-10-2024","place":"Chennai"},
            # {"temperature":"51","Time":"30-11-2024","place":"Coimbatore"},
            # {"temperature":"55","Time":"8-12-2026","place":"Trichy"},
            # {"temperature":"40","Time":"13-1-2026","place":"Salem"},
            # {"temperature":"31","Time":"25-2-2026","place":"Erode"},
            # {"temperature":"45","Time":"9-3-2024","place":"Tirunelveli"},
            # {"temperature":"50","Time":"17-4-2025","place":"Madurai"},
            # {"temperature":"30","Time":"2-5-2025","place":"Chennai"},

            # {"temperature":"29","Time":"13-6-2025","place":"Coimbatore"},
            # {"temperature":"44","Time":"21-7-2025","place":"Trichy"},
            # {"temperature":"42","Time":"30-8-2025","place":"Salem"},
            # {"temperature":"37","Time":"10-9-2025","place":"Erode"},
            # {"temperature":"33","Time":"23-10-2025","place":"Tirunelveli"},
            # {"temperature":"49","Time":"6-11-2025","place":"Madurai"},
            # {"temperature":"34","Time":"17-12-2025","place":"Chennai"},
            # {"temperature":"45","Time":"29-1-2026","place":"Coimbatore"},
            # {"temperature":"38","Time":"14-2-2026","place":"Trichy"},
            # {"temperature":"36","Time":"28-3-2026","place":"Salem"}
        ]
    }
}


    print("=== ORIGINAL ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    save_json(payload, "original.json")

    encoded = encode(payload, value_min_freq=2, compact=True)
    print("\n=== ENCODED (send to LLM) ===")
    print(json.dumps(encoded, indent=2, ensure_ascii=False))
    save_json(encoded, "encoded.json")

    # simulate LLM echo
    received = encoded
    decoded = decode(received)
    print("\n=== DECODED ===")
    print(json.dumps(decoded, indent=2, ensure_ascii=False))
    save_json(decoded, "decoded.json")

    orig_s = file_stats(Path("original.json"))
    enc_s = file_stats(Path("encoded.json"))
    dec_s = file_stats(Path("decoded.json"))

    print("\n=== FILE STATS ===")
    for s in (orig_s, enc_s, dec_s):
        print(f"{s['path']}: {s['bytes']} bytes, {s['chars']} chars, ~{s['tokens']} tokens")

    print("\n=== DATA FIELD STATS ===")
    od = data_text(payload)
    ed = data_text(encoded)
    dd = data_text(decoded)
    print(f"original.data: {len(od.encode('utf8'))} bytes, {len(od)} chars, ~{token_count(od)} tokens")
    print(f"encoded.data:  {len(ed.encode('utf8'))} bytes, {len(ed)} chars, ~{token_count(ed)} tokens")
    print(f"decoded.data:  {len(dd.encode('utf8'))} bytes, {len(dd)} chars, ~{token_count(dd)} tokens")

    def pct(a,b):
        if a==0:
            return None
        return round((1 - (b/a)) * 100, 2)

    print("\n=== COMPARISONS ===")
    print(f"Total tokens: original {orig_s['tokens']} -> encoded {enc_s['tokens']}  | savings: {pct(orig_s['tokens'], enc_s['tokens'])}%")
    print(f"Data-only tokens: original {token_count(od)} -> encoded {token_count(ed)}  | savings: {pct(token_count(od), token_count(ed))}%")
    print(f"bytes: original {orig_s['bytes']} -> encoded {enc_s['bytes']} | savings: {pct(orig_s['bytes'], enc_s['bytes'])}%")

    summary = {
        "original": orig_s, "encoded": enc_s, "decoded": dec_s,
        "original_data": {"chars": len(od), "bytes": len(od.encode("utf8")), "tokens": token_count(od)},
        "encoded_data":  {"chars": len(ed), "bytes": len(ed.encode("utf8")), "tokens": token_count(ed)},
        "comparisons": {
            "tokens_total_savings_pct": pct(orig_s['tokens'], enc_s['tokens']),
            "tokens_data_savings_pct": pct(token_count(od), token_count(ed)),
            "bytes_savings_pct": pct(orig_s['bytes'], enc_s['bytes'])
        },
        "note": f"tokenizer={MODEL}" if TIK else "tokenizer not available (heuristic used)"
    }
    Path("token_stats_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
    print("\nWrote token_stats_summary.json")

if __name__ == "__main__":
    demo()
