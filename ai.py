import json
import math
import time
import subprocess


# ----------------------------------------------------------
# 1) Run prompt through local Ollama with STREAMING
# ----------------------------------------------------------
def ollama_infer(model: str, prompt: str, timeout_sec=200):
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        process.stdin.write(prompt.encode("utf-8"))
        process.stdin.close()

        start = time.time()
        output = ""

        # stream output line by line
        while True:
            if process.poll() is not None:
                break

            line = process.stdout.readline().decode("utf-8", errors="ignore")
            if line:
                output += line

            if time.time() - start > timeout_sec:
                process.kill()
                return "[TIMEOUT] Model exceeded timeout"

        return output.strip()

    except Exception as e:
        return f"[ERROR] {str(e)}"


# ----------------------------------------------------------
# 2) Token estimator for Gemma
# ----------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 3.5))



# ----------------------------------------------------------
# 3) Build JSON-only evaluation prompt
# ----------------------------------------------------------
def build_understanding_prompt(data_text: str):
    return f"""
You are a strict evaluator.

Analyze the dataset below and respond ONLY with JSON in this format:
{{
  "understanding": <0-100>,
  "confidence": <0-100>,
  "ease": <0-100>,
  "ambiguous": "yes" or "no",
  "explanation": "<short reason>"
}}

DATA:
{
  "device": "sensor-xyz",
  "area": "Tamilnadu",
  "data": {
    "meta": "META&ORDER=Time,place,temperature&vmap=V1:Tirunelveli;V3:Coimbatore",
    "body": "BODY|sensordata[30]{Time,place,temperature}|3-8-2025,Madurai,49|5-9-2025,Chennai,35|2-3-2025,Madurai,35|11-1-2024,V3,42|14-2-2024,Trichy,39|9-3-2024,Salem,51|28-4-2025,Erode,47|15-5-2025,V1,33|7-6-2023,Madurai,55|21-7-2023,Chennai,29|10-8-2023,V3,38|3-9-2023,Trichy,41|19-10-2024,Salem,36|26-11-2026,Erode,44|5-12-2026,V1,32|6-1-2024,Madurai,28|17-2-2025,Chennai,53|25-3-2025,V3,37|2-4-2025,Trichy,54|14-5-2023,Salem,46|18-6-2023,Erode,33|9-7-2023,V1,45|27-8-2024,Madurai,31|15-9-2024,Chennai,52|30-10-2025,V3,40|4-11-2025,Trichy,34|12-12-2025,Salem,48|19-1-2026,Erode,50|23-2-2026,V1,43|3-3-2023,Madurai,29"
  }
}
"""


# ----------------------------------------------------------
# 4) Get Model Scores
# ----------------------------------------------------------
def get_llm_understanding_scores(model: str, data_str: str):
    prompt = build_understanding_prompt(data_str)

    start = time.time()
    response = ollama_infer(model, prompt)
    latency = round(time.time() - start, 3)

    # Try parsing JSON
    try:
        first = response.find("{")
        last = response.rfind("}")
        cleaned = response[first:last+1]
        report = json.loads(cleaned)
    except Exception:
        report = {"error": "parse_failed", "raw": response}

    report["latency_sec"] = latency
    return report



# ----------------------------------------------------------
# 5) Main pipeline
# ----------------------------------------------------------
def run_analysis(original_obj, encoded_obj, model="gemma:2b"):
    original_str = json.dumps(original_obj, ensure_ascii=False)
    encoded_str  = json.dumps(encoded_obj, ensure_ascii=False)

    tok_orig = estimate_tokens(original_str)
    tok_enc  = estimate_tokens(encoded_str)

    print("\n=== TOKEN STATISTICS ===")
    print(f"Original tokens : {tok_orig}")
    print(f"Encoded tokens  : {tok_enc}")
    print(f"Savings         : {round((1 - tok_enc/tok_orig) * 100, 2)}%")

    print("\nRunning model understanding tests on LOCAL Gemma 2B...\n")

    # ORIGINAL
    print("=== MODEL UNDERSTANDING: ORIGINAL JSON ===")
    orig_report = get_llm_understanding_scores(model, original_str)
    print(json.dumps(orig_report, indent=2, ensure_ascii=False))

    # ENCODED
    print("\n=== MODEL UNDERSTANDING: COIL-ENCODED JSON ===")
    enc_report = get_llm_understanding_scores(model, encoded_str)
    print(json.dumps(enc_report, indent=2, ensure_ascii=False))

    return {
        "token_stats": {
            "original_tokens": tok_orig,
            "encoded_tokens": tok_enc,
            "savings_pct": round((1 - tok_enc/tok_orig) * 100, 2)
        },
        "model_reports": {
            "original": orig_report,
            "encoded": enc_report
        }
    }



# ----------------------------------------------------------
# 6) Execute
# ----------------------------------------------------------
if __name__ == "__main__":
    with open("original.json", "r", encoding="utf8") as f:
        orig = json.load(f)

    with open("encoded.json", "r", encoding="utf8") as f:
        enc = json.load(f)

    result = run_analysis(orig, enc)

    with open("ollama_interpretability_report.json", "w", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nSaved: ollama_interpretability_report.json")
