"""JSON validation and schema compliance metrics for structured extraction task."""
import json
import re
import numpy as np

EXPECTED_FIELDS = ["objective", "method", "key_result", "model_or_system", "benchmark"]

# Regex to extract JSON object from text that may contain preamble
_JSON_OBJECT_RE = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)


def _try_parse_json(text: str):
    """Try to parse JSON directly, then try extracting from preamble text.

    Returns (parsed_dict_or_None, raw_valid_bool, extracted_valid_bool).
    """
    if not text or not isinstance(text, str):
        return None, False, False

    # Try direct parse first
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed, True, True
        return None, False, False
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON object from text with preamble
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed, False, True
        except (json.JSONDecodeError, TypeError):
            pass

    return None, False, False


def json_validity_rate(outputs: list) -> dict:
    """Check what fraction of outputs are valid JSON (raw and extracted)."""
    raw_valid = 0
    extracted_valid = 0
    for out in outputs:
        _, is_raw, is_extracted = _try_parse_json(out)
        if is_raw:
            raw_valid += 1
        if is_extracted:
            extracted_valid += 1
    n = len(outputs) if outputs else 1
    return {
        "json_valid_raw_count": raw_valid,
        "json_valid_extracted_count": extracted_valid,
        "json_valid_total": len(outputs),
        "json_validity_rate_raw": raw_valid / n,
        "json_validity_rate_extracted": extracted_valid / n,
        # Keep backward-compatible key (now uses extracted)
        "json_validity_rate": extracted_valid / n,
    }


def schema_compliance_rate(outputs: list) -> dict:
    """Check what fraction of parseable JSON outputs have all expected fields."""
    compliant = 0
    parseable = 0
    for out in outputs:
        parsed, _, is_extracted = _try_parse_json(out)
        if parsed is not None and is_extracted:
            parseable += 1
            if all(f in parsed for f in EXPECTED_FIELDS):
                compliant += 1
    n = len(outputs) if outputs else 1
    return {
        "schema_compliant_count": compliant,
        "schema_compliant_of_valid": compliant / parseable if parseable > 0 else 0.0,
        "schema_compliance_rate": compliant / n,
    }


def field_level_accuracy(outputs: list) -> dict:
    """Compute pairwise field-level exact match for all output pairs."""
    import itertools

    parsed_outputs = []
    for out in outputs:
        parsed, _, _ = _try_parse_json(out)
        if parsed is not None:
            parsed_outputs.append(parsed)

    if len(parsed_outputs) < 2:
        return {"field_accuracy": {f: None for f in EXPECTED_FIELDS}, "overall_field_emr": None}

    pairs = list(itertools.combinations(range(len(parsed_outputs)), 2))
    field_matches = {f: [] for f in EXPECTED_FIELDS}

    for i, j in pairs:
        for field in EXPECTED_FIELDS:
            val_i = (parsed_outputs[i].get(field) or "").strip()
            val_j = (parsed_outputs[j].get(field) or "").strip()
            field_matches[field].append(1.0 if val_i == val_j else 0.0)

    field_accuracy = {f: float(np.mean(field_matches[f])) for f in EXPECTED_FIELDS}
    overall = float(np.mean([v for vals in field_matches.values() for v in vals]))

    return {
        "field_accuracy": field_accuracy,
        "overall_field_emr": overall
    }
