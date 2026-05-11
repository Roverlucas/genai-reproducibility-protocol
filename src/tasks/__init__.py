"""Task loaders and metric implementations for the NatComms revision.

Submodules:
    humaneval_loader: HumanEval (OpenAI) code generation problems.
    gsm8k_loader:     GSM8K (Cobbe et al.) grade-school math problems.
    pubmed_loader:    PM2.5/respiratory PubMed abstracts (T14).
    pass_at_1:        Sandboxed code execution + pass@1 metric.
    gsm8k_extractor:  Final-answer parsing for GSM8K.

These modules are scaffolding only — they support the unified runner in
`run_revision_experiments.py` but make no API calls themselves.
"""
