"""W3C PROV-JSON generator for the GenAI reproducibility protocol.

Generates provenance graphs compliant with the W3C PROV data model,
representing experimental runs as activities that use and generate
entities, associated with agents.
"""

import json
from pathlib import Path
from typing import Optional


NAMESPACE_PREFIX = "genai"
NAMESPACE_URI = "https://genai-prov.org/ns#"
PROV_URI = "http://www.w3.org/ns/prov#"


class ProvGenerator:
    """Generates PROV-JSON provenance documents from run data."""

    def __init__(self, output_dir: str = "outputs/prov"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_run(self, run_data: dict) -> dict:
        """Generate a PROV-JSON document from a single run record."""
        run_id = run_data["run_id"]
        ns = NAMESPACE_PREFIX

        prov_doc = {
            "prefix": {
                "prov": PROV_URI,
                ns: NAMESPACE_URI,
            },
            "entity": {},
            "activity": {},
            "agent": {},
            "wasGeneratedBy": {},
            "used": {},
            "wasAssociatedWith": {},
            "wasAttributedTo": {},
            "wasDerivedFrom": {},
        }

        # --- Entities ---

        # Prompt entity
        prompt_eid = f"{ns}:prompt_{run_data['prompt_hash'][:12]}"
        prov_doc["entity"][prompt_eid] = {
            "prov:label": "Prompt Artifact",
            "prov:type": f"{ns}:Prompt",
            f"{ns}:prompt_hash": run_data["prompt_hash"],
            f"{ns}:task_category": run_data["task_category"],
        }

        # Model entity
        model_eid = f"{ns}:model_{run_data['model_name']}_{run_data['model_version']}"
        model_eid = model_eid.replace(" ", "_").replace(".", "_")
        prov_doc["entity"][model_eid] = {
            "prov:label": "ModelVersion Artifact",
            "prov:type": f"{ns}:ModelVersion",
            f"{ns}:model_name": run_data["model_name"],
            f"{ns}:model_version": run_data["model_version"],
            f"{ns}:weights_hash": run_data.get("weights_hash", ""),
            f"{ns}:model_source": run_data.get("model_source", ""),
        }

        # Inference parameters entity
        params_eid = f"{ns}:params_{run_data['params_hash'][:12]}"
        prov_doc["entity"][params_eid] = {
            "prov:label": "InferenceParameters Artifact",
            "prov:type": f"{ns}:InferenceParameters",
        }
        for k, v in run_data.get("inference_params", {}).items():
            prov_doc["entity"][params_eid][f"{ns}:{k}"] = str(v)

        # Output entity
        output_eid = f"{ns}:{run_id}_output"
        prov_doc["entity"][output_eid] = {
            "prov:label": "Output Artifact",
            "prov:type": f"{ns}:Output",
            f"{ns}:output_hash": run_data.get("output_hash", ""),
            f"{ns}:timestamp_generated": run_data.get("timestamp_end", ""),
        }
        if run_data.get("output_metrics"):
            for k, v in run_data["output_metrics"].items():
                prov_doc["entity"][output_eid][f"{ns}:metric_{k}"] = str(v)

        # Execution metadata entity
        exec_eid = f"{ns}:{run_id}_metadata"
        prov_doc["entity"][exec_eid] = {
            "prov:label": "ExecutionMetadata",
            "prov:type": f"{ns}:ExecutionMetadata",
            f"{ns}:environment_hash": run_data.get("environment_hash", ""),
            f"{ns}:code_commit": run_data.get("code_commit", ""),
            f"{ns}:execution_duration_ms": str(
                run_data.get("execution_duration_ms", "")
            ),
        }

        # Context corpus entity (if RAG)
        if run_data.get("retrieval_context"):
            ctx = run_data["retrieval_context"]
            ctx_eid = f"{ns}:context_{ctx.get('corpus_id', 'default')}"
            prov_doc["entity"][ctx_eid] = {
                "prov:label": "ContextCorpus Artifact",
                "prov:type": f"{ns}:ContextCorpus",
                f"{ns}:corpus_id": ctx.get("corpus_id", ""),
                f"{ns}:index_version": ctx.get("index_version", ""),
            }

        # Input text entity (if provided)
        if run_data.get("input_hash"):
            input_eid = f"{ns}:input_{run_data['input_hash'][:12]}"
            prov_doc["entity"][input_eid] = {
                "prov:label": "Input Text",
                "prov:type": f"{ns}:InputText",
                f"{ns}:input_hash": run_data["input_hash"],
            }

        # --- Activity ---
        activity_id = f"{ns}:{run_id}"
        prov_doc["activity"][activity_id] = {
            "prov:label": "RunGeneration Activity",
            "prov:type": f"{ns}:RunGeneration",
            "prov:startTime": run_data.get("timestamp_start", ""),
            "prov:endTime": run_data.get("timestamp_end", ""),
        }

        # --- Agents ---
        researcher_aid = f"{ns}:researcher_{run_data.get('researcher_id', 'anon')}"
        prov_doc["agent"][researcher_aid] = {
            "prov:type": "prov:Person",
            "prov:label": "Researcher",
            f"{ns}:researcher_id": run_data.get("researcher_id", ""),
            f"{ns}:affiliation": run_data.get("affiliation", ""),
        }

        system_aid = f"{ns}:system_{run_data.get('environment_hash', 'unknown')[:8]}"
        prov_doc["agent"][system_aid] = {
            "prov:type": "prov:SoftwareAgent",
            "prov:label": "SystemExecutor",
            f"{ns}:environment_hash": run_data.get("environment_hash", ""),
        }

        # --- Relations ---
        # wasGeneratedBy
        prov_doc["wasGeneratedBy"][f"_:wgb_{run_id}"] = {
            "prov:entity": output_eid,
            "prov:activity": activity_id,
        }

        # used
        use_idx = 1
        prov_doc["used"][f"_:u{use_idx}_{run_id}"] = {
            "prov:activity": activity_id,
            "prov:entity": prompt_eid,
        }
        use_idx += 1
        prov_doc["used"][f"_:u{use_idx}_{run_id}"] = {
            "prov:activity": activity_id,
            "prov:entity": model_eid,
        }
        use_idx += 1
        prov_doc["used"][f"_:u{use_idx}_{run_id}"] = {
            "prov:activity": activity_id,
            "prov:entity": params_eid,
        }
        if run_data.get("input_hash"):
            use_idx += 1
            prov_doc["used"][f"_:u{use_idx}_{run_id}"] = {
                "prov:activity": activity_id,
                "prov:entity": input_eid,
            }
        if run_data.get("retrieval_context"):
            use_idx += 1
            prov_doc["used"][f"_:u{use_idx}_{run_id}"] = {
                "prov:activity": activity_id,
                "prov:entity": ctx_eid,
            }

        # wasAssociatedWith
        prov_doc["wasAssociatedWith"][f"_:waw_{run_id}"] = {
            "prov:activity": activity_id,
            "prov:agent": researcher_aid,
        }

        # wasAttributedTo
        prov_doc["wasAttributedTo"][f"_:wat_{run_id}"] = {
            "prov:entity": output_eid,
            "prov:agent": system_aid,
        }

        # wasDerivedFrom
        prov_doc["wasDerivedFrom"][f"_:wdf_{run_id}"] = {
            "prov:generatedEntity": output_eid,
            "prov:usedEntity": prompt_eid,
        }

        return prov_doc

    def generate_multi_run(self, run_data_list: list) -> dict:
        """Generate a merged PROV-JSON document from multiple runs."""
        merged = {
            "prefix": {"prov": PROV_URI, NAMESPACE_PREFIX: NAMESPACE_URI},
            "entity": {},
            "activity": {},
            "agent": {},
            "wasGeneratedBy": {},
            "used": {},
            "wasAssociatedWith": {},
            "wasAttributedTo": {},
            "wasDerivedFrom": {},
        }

        for run_data in run_data_list:
            single = self.generate_from_run(run_data)
            for section in [
                "entity",
                "activity",
                "agent",
                "wasGeneratedBy",
                "used",
                "wasAssociatedWith",
                "wasAttributedTo",
                "wasDerivedFrom",
            ]:
                merged[section].update(single.get(section, {}))

        return merged

    def save(self, prov_doc: dict, filename: str = "provenance") -> str:
        """Save a PROV-JSON document. Returns filepath."""
        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(prov_doc, f, indent=2, ensure_ascii=False)
        return str(filepath)
