"""JSON validation and schema compliance metrics for structured extraction task."""
import json
import numpy as np

EXPECTED_FIELDS = ["objective", "method", "key_result", "model_or_system", "benchmark"]


def json_validity_rate(outputs: list) -> dict:
    """Check what fraction of outputs are valid JSON."""
    valid = 0
    for out in outputs:
        try:
            json.loads(out)
            valid += 1
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "json_valid_count": valid,
        "json_valid_total": len(outputs),
        "json_validity_rate": valid / len(outputs) if outputs else 0.0
    }


def schema_compliance_rate(outputs: list) -> dict:
    """Check what fraction of valid JSON outputs have all expected fields."""
    compliant = 0
    valid = 0
    for out in outputs:
        try:
            parsed = json.loads(out)
            valid += 1
            if isinstance(parsed, dict) and all(f in parsed for f in EXPECTED_FIELDS):
                compliant += 1
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "schema_compliant_count": compliant,
        "schema_compliant_of_valid": compliant / valid if valid > 0 else 0.0,
        "schema_compliance_rate": compliant / len(outputs) if outputs else 0.0
    }


def field_level_accuracy(outputs: list) -> dict:
    """Compute pairwise field-level exact match for all output pairs."""
    import itertools

    parsed_outputs = []
    for out in outputs:
        try:
            p = json.loads(out)
            if isinstance(p, dict):
                parsed_outputs.append(p)
        except (json.JSONDecodeError, TypeError):
            pass

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
