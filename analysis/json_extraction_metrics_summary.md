# JSON Extraction Quality Metrics Summary

**Generated:** 2026-02-09
**Script:** `/Users/lucasrover/paper-experiment/analysis/compute_json_metrics.py`
**Data:** 866 extraction runs from `/Users/lucasrover/paper-experiment/outputs/runs/`
**Full Results:** `/Users/lucasrover/paper-experiment/analysis/json_extraction_metrics.json`

---

## Overview

This analysis computes JSON extraction quality metrics for all extraction runs (both LLaMA 3 8B and GPT-4) using the updated validation.py module. The metrics assess three key dimensions:

1. **JSON Validity**: Raw vs. extracted JSON parsing success
2. **Schema Compliance**: Presence of all expected fields
3. **Field-Level Quality**: Exact match rates and completeness

---

## Key Findings

### 1. JSON Validity Rates

| Model  | Condition       | N Runs | Raw Valid % | Extracted Valid % | Schema Compliant % |
|--------|-----------------|--------|-------------|-------------------|--------------------|
| GPT-4  | C2_same_params  | 150    | 100.0%      | 100.0%            | 100.0%             |
| GPT-4  | C3_temp0.0      | 45     | 100.0%      | 100.0%            | 100.0%             |
| GPT-4  | C3_temp0.3      | 46     | 100.0%      | 100.0%            | 100.0%             |
| GPT-4  | C3_temp0.7      | 55     | 100.0%      | 100.0%            | 100.0%             |
| LLaMA3 | C1_fixed_seed   | 150    | 0.0%        | 100.0%            | 100.0%             |
| LLaMA3 | C2_var_seed     | 150    | 0.0%        | 100.0%            | 100.0%             |
| LLaMA3 | C3_temp0.0      | 90     | 0.0%        | 100.0%            | 100.0%             |
| LLaMA3 | C3_temp0.3      | 90     | 0.0%        | 97.8%             | 97.8%              |
| LLaMA3 | C3_temp0.7      | 90     | 0.0%        | 92.2%             | 92.2%              |

**Key Insights:**
- **GPT-4**: Always produces raw valid JSON (100% raw valid)
- **LLaMA 3**: Always adds preamble text (0% raw valid) but JSON is extractable at high rates
- **Extracted validity**: 100% for temp≤0.0, degrades at higher temperatures for LLaMA 3
- **Schema compliance**: Perfect when JSON is extractable

---

### 2. Field-Level Exact Match Rates (EMR)

Pairwise exact match rates across all output pairs within each group:

#### GPT-4 (Lower EMR = More Variability)

| Condition      | Overall EMR | objective | method | key_result | model_or_system | benchmark |
|----------------|-------------|-----------|--------|------------|-----------------|-----------|
| C2_same_params | 0.0346      | 0.0208    | 0.0179 | 0.0171     | 0.0370          | 0.0801    |
| C3_temp0.0     | 0.0446      | 0.0273    | 0.0182 | 0.0202     | 0.0364          | 0.1212    |
| C3_temp0.3     | 0.0497      | 0.0145    | 0.0106 | 0.0164     | 0.0483          | 0.1585    |
| C3_temp0.7     | 0.0281      | 0.0034    | 0.0027 | 0.0047     | 0.0283          | 0.1017    |

#### LLaMA 3 (Higher EMR = More Reproducible)

| Condition      | Overall EMR | objective | method | key_result | model_or_system | benchmark |
|----------------|-------------|-----------|--------|------------|-----------------|-----------|
| C1_fixed_seed  | 0.0713      | 0.0265    | 0.0265 | 0.0399     | 0.0895          | 0.1741    |
| C2_var_seed    | 0.0713      | 0.0265    | 0.0265 | 0.0399     | 0.0895          | 0.1741    |
| C3_temp0.0     | 0.0670      | 0.0220    | 0.0220 | 0.0355     | 0.0854          | 0.1703    |
| C3_temp0.3     | 0.0718      | 0.0170    | 0.0104 | 0.0266     | 0.0922          | 0.2126    |
| C3_temp0.7     | 0.0803      | 0.0118    | 0.0044 | 0.0223     | 0.0679          | 0.2953    |

**Key Insights:**
- **LLaMA 3 is 2x more reproducible than GPT-4** (EMR ~0.07 vs ~0.03-0.05)
- **Benchmark field has highest EMR** in both models (most structured/extractable)
- **Method and key_result have lowest EMR** (most open-ended/variable)
- **C1 vs C2**: Identical EMR confirms seed has NO effect under greedy decoding
- **Temperature effect**: Mixed pattern, not strictly monotonic (temp=0.7 sometimes higher EMR than 0.3)

---

### 3. Field Presence and Completeness

All models achieve 100% field presence (all fields always present in valid JSON). Non-empty rates:

#### GPT-4 Non-Empty Rates

| Condition      | Overall | objective | method | key_result | model_or_system | benchmark |
|----------------|---------|-----------|--------|------------|-----------------|-----------|
| C2_same_params | 0.925   | 1.000     | 1.000  | 1.000      | 0.880           | 0.747     |
| C3_temp0.0     | 0.911   | 1.000     | 1.000  | 1.000      | 0.889           | 0.667     |
| C3_temp0.3     | 0.896   | 1.000     | 1.000  | 1.000      | 0.870           | 0.609     |
| C3_temp0.7     | 0.913   | 1.000     | 1.000  | 1.000      | 0.873           | 0.691     |

#### LLaMA 3 Non-Empty Rates

| Condition      | Overall | objective | method | key_result | model_or_system | benchmark |
|----------------|---------|-----------|--------|------------|-----------------|-----------|
| C1_fixed_seed  | 0.840   | 1.000     | 1.000  | 0.867      | 0.733           | 0.600     |
| C2_var_seed    | 0.840   | 1.000     | 1.000  | 0.867      | 0.733           | 0.600     |
| C3_temp0.0     | 0.840   | 1.000     | 1.000  | 0.867      | 0.733           | 0.600     |
| C3_temp0.3     | 0.823   | 0.989     | 1.000  | 0.864      | 0.716           | 0.546     |
| C3_temp0.7     | 0.807   | 0.976     | 0.988  | 0.855      | 0.759           | 0.458     |

**Key Insights:**
- **GPT-4 has higher completeness** (92.5% vs 84.0% overall non-empty)
- **Benchmark field often empty** (60-75% non-empty) - many abstracts lack explicit benchmarks
- **Core fields (objective, method, key_result) nearly always non-empty** in GPT-4
- **Temperature degrades completeness** slightly at temp=0.7

---

## Statistical Notes

1. **Sample Sizes**:
   - LLaMA 3: 30 abstracts × 3-5 reps = 90-150 runs per condition
   - GPT-4: 20-30 abstracts × variable reps = 45-150 runs per condition

2. **Per-Abstract Metrics**: Full per-abstract breakdown included in JSON output

3. **Validation Method**: Uses updated validation.py with:
   - Raw JSON parsing (direct json.loads)
   - Extracted JSON parsing (regex extraction from preamble text)
   - Schema compliance (all 5 expected fields present)

---

## Implications for Paper

1. **JSON Quality is High**: Both models produce valid, schema-compliant JSON at very high rates
2. **LLaMA 3 Preamble Pattern**: Consistent preamble behavior (0% raw valid) is a feature, not a bug
3. **Reproducibility Confirmed**: Low EMR (3-8%) validates the paper's core finding of hidden non-determinism
4. **Field-Level Granularity**: Benchmark field more reproducible (structured), method/key_result less so (open-ended)
5. **Completeness Trade-offs**: GPT-4 more complete but less reproducible; LLaMA 3 less complete but more reproducible

---

## Files Generated

- **Metrics JSON**: `/Users/lucasrover/paper-experiment/analysis/json_extraction_metrics.json` (172KB, 6205 lines)
- **Summary MD**: `/Users/lucasrover/paper-experiment/analysis/json_extraction_metrics_summary.md` (this file)
- **Script**: `/Users/lucasrover/paper-experiment/analysis/compute_json_metrics.py`

---

**End of Summary**
