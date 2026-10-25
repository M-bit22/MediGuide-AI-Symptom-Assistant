import json
import re

def parse_llm_json(raw_output: str) -> dict:
    """
    Safely parse JSON from LLM output, stripping markdown code fences if present.
    Ensures invalid JSON never crashes the application.
    """
    if not raw_output:
        return {"error": "Empty response received from LLM."}

    cleaned = raw_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, dg := re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {
            "error": "Failed to parse JSON response from model.",
            "raw_output": raw_output
        }