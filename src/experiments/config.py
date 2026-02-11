"""Experiment configuration for the GenAI reproducibility study."""

# Researcher metadata
RESEARCHER_ID = "lucas_rover"
AFFILIATION = "UTFPR - Universidade Tecnologica Federal do Parana"

# Model configurations
LLAMA_MODEL = "llama3:8b"
GPT4_MODEL = "gpt-4"
MISTRAL_MODEL = "mistral:7b"
GEMMA2_MODEL = "gemma2:9b"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "gemini-2.5-pro"
DEEPSEEK_MODEL = "deepseek-chat"
PERPLEXITY_MODEL = "sonar"
TOGETHER_LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

# All local models (served via Ollama)
LOCAL_MODELS = [LLAMA_MODEL, MISTRAL_MODEL, GEMMA2_MODEL]

# API models
API_MODELS = [GPT4_MODEL, CLAUDE_MODEL, GEMINI_MODEL, DEEPSEEK_MODEL, PERPLEXITY_MODEL]

# Cloud-served open-weight models (for architecture vs infrastructure isolation)
CLOUD_OPEN_WEIGHT_MODELS = [TOGETHER_LLAMA_MODEL]

# Number of repetitions per condition
N_REPS = 5

# Seeds for controlled experiments
SEEDS = [42, 123, 456, 789, 1024]

# Temperature variations
TEMPERATURES = [0.0, 0.3, 0.7]

# Prompts
SUMMARIZATION_PROMPT = (
    "You are a scientific summarization assistant. "
    "Read the following scientific abstract and produce a concise summary "
    "in exactly 3 sentences. The summary must: "
    "(1) state the main contribution or finding, "
    "(2) describe the methodology used, and "
    "(3) report the key quantitative result if available. "
    "Do not add any information not present in the original abstract. "
    "Do not include any preamble or explanation — output only the 3-sentence summary."
)

EXTRACTION_PROMPT = (
    "You are a structured information extraction assistant. "
    "Read the following scientific abstract and extract the information "
    "into the exact JSON format below. Use only information explicitly stated "
    "in the abstract. If a field is not mentioned, use null.\n\n"
    "Output format (JSON only, no explanation):\n"
    "{\n"
    '  "objective": "string — main goal of the study",\n'
    '  "method": "string — methodology or approach used",\n'
    '  "key_result": "string — most important quantitative or qualitative result",\n'
    '  "model_or_system": "string — name of the model/system proposed (if any)",\n'
    '  "benchmark": "string — evaluation benchmark or dataset used (if any)"\n'
    "}"
)
