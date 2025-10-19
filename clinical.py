#!/usr/bin/env python3
"""
Upgraded Clinical Co-Pilot Agent (single-file agentic demo)

This version has been updated to run a local Med-Gemma model, replacing the remote API call.
- Loads a local, multimodal Med-Gemma model (`google/medgemma-4b-it`) using transformers.
- Implements a simple RAG pipeline with an in-memory knowledge base.
- Includes a planner that decomposes goals into subtasks with risk-aware execution policies.
- Simulates production actions like patching EHR records via a FHIR client.
- Persists analyses and audit logs in SQLite.

NOTES:
- This is a self-contained demo. You must install dependencies:
  pip install torch pillow accelerate "transformers>=4.40" requests numpy
- The first time you run this, it will download the Med-Gemma model (approx. 9GB).
- Replace placeholder secrets and endpoints before using in a real environment.
- Do not use this with real PHI until you implement security, access controls,
  encryption, and run in a compliant environment.
"""

import sqlite3
import json
import uuid
import datetime
import asyncio
import logging
import requests
from typing import Any, Dict, List, Optional
from difflib import SequenceMatcher

# --- New Imports for Local Model Inference ---
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
except ImportError:
    print("Warning: Required packages not found.")
    print("Please run: pip install torch pillow accelerate transformers requests")
    torch = Image = AutoProcessor = AutoModelForImageTextToText = None


# ---------------------------
# Configuration
# ---------------------------
DB_FILE = "clinical_copilot_agent.db"
AUTO_EXECUTE_RISK_LEVEL = 1  # <= this value will auto-execute; higher requires approval
# risk levels: 0=info-only, 1=low risk (scheduling), 2=clinical orders, 3=emergency

# --- Configuration for Local Model ---
MODEL_ID = "google/medgemma-4b-it"

# 3. EHR Integration
EHR_BASE = "https://ehr.internal/fhir"  # replace with real FHIR server
EHR_TOKEN = "__REPLACE_EHR_TOKEN__"  # store in KMS / Vault

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("clinical_copilot")

# ---------------------------
# Local Model Loading
# ---------------------------
MODEL = None
PROCESSOR = None

def load_medgemma_model():
    """Loads the Med-Gemma model and processor into global variables."""
    global MODEL, PROCESSOR
    if not all([torch, Image, AutoProcessor, AutoModelForImageTextToText]):
        logger.error("Required AI/ML libraries not found. Cannot load model.")
        return

    if MODEL is None:
        try:
            logger.info(f"Loading Med-Gemma model ({MODEL_ID}). This may take a while and download several GBs...")
            MODEL = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
            logger.info("Med-Gemma model loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load Med-Gemma model: {e}")
            MODEL = PROCESSOR = None # Ensure it's unset on failure

# ---------------------------
# Persistence / DB (No changes)
# ---------------------------
def init_db(db_file: str = DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, patient_id TEXT, narrative TEXT, medgemma_output TEXT,
            soap_note TEXT, icd_tags TEXT, status TEXT, created_at TEXT, updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT, event TEXT,
            details TEXT, timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn

DB = init_db()

def write_analysis(analysis: Dict[str, Any]):
    cur = DB.cursor()
    cur.execute(
        "INSERT INTO analyses (id, patient_id, narrative, medgemma_output, soap_note, icd_tags, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            analysis["id"], analysis["patient_id"], analysis["narrative"],
            json.dumps(analysis.get("medgemma_output")), analysis.get("soap_note"),
            json.dumps(analysis.get("icd_tags")), analysis.get("status", "queued"),
            analysis.get("created_at"), analysis.get("updated_at"),
        ),
    )
    DB.commit()

def update_analysis(analysis_id: str, updates: Dict[str, Any]):
    cur = DB.cursor()
    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
    params = list(updates.values())
    cur.execute(f"UPDATE analyses SET {set_clause}, updated_at = ? WHERE id = ?", params + [datetime.datetime.utcnow().isoformat(), analysis_id])
    DB.commit()

def log_audit(analysis_id: str, event: str, details: Any):
    cur = DB.cursor()
    cur.execute(
        "INSERT INTO audit_logs (analysis_id, event, details, timestamp) VALUES (?, ?, ?, ?)",
        (analysis_id, event, json.dumps(details), datetime.datetime.utcnow().isoformat())
    )
    DB.commit()

# ---------------------------
# Utilities: FHIR -> Narrative (No changes)
# ---------------------------
def fhir_to_narrative(fhir_bundle: Dict[str, Any]) -> str:
    parts = []
    patient = next((e["resource"] for e in fhir_bundle.get("entry", []) if e.get("resource", {}).get("resourceType") == "Patient"), None)
    if patient:
        name = patient.get("name", [{}])[0].get("text", "Unknown")
        parts.append(f"Patient: {name}, gender={patient.get('gender','?')}, dob={patient.get('birthDate','?')}")
    obs = [e["resource"] for e in fhir_bundle.get("entry", []) if e.get("resource", {}).get("resourceType") == "Observation"]
    if obs:
        parts.append("Observations:")
        for o in obs:
            code, val, date = o.get("code", {}).get("text", "obs"), o.get("valueQuantity", {}).get("value", ""), o.get("effectiveDateTime", "")
            parts.append(f"- {code}: {val} ({date})")
    conds = [e["resource"] for e in fhir_bundle.get("entry", []) if e.get("resource", {}).get("resourceType") == "Condition"]
    if conds:
        parts.append("Problems:")
        for c in conds:
            parts.append(f"- {c.get('code', {}).get('text', 'condition')}")
    return "\n".join(parts) if parts else "No narrative data."

# ---------------------------
# RAG: Simple Knowledge Base
# ---------------------------
class SimpleKnowledgeBase:
    """A minimal, in-memory 'knowledge base' and naive retriever using substring/similarity."""
    def __init__(self):
        self.docs: List[Dict[str, str]] = []

    def add_doc(self, doc_id: str, text: str, meta: Optional[Dict] = None):
        self.docs.append({"id": doc_id, "text": text, "meta": meta or {}})

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        scored = []
        for d in self.docs:
            score = SequenceMatcher(None, query.lower(), d["text"].lower()).ratio()
            if query.lower() in d["text"].lower(): score = 1.0
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"id": d["id"], "text": d["text"], "score": s} for s, d in scored[:top_k]]

def rag_retrieve(kb: SimpleKnowledgeBase, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    return kb.retrieve(query, top_k=top_k)

# ---------------------------
# Local Med-Gemma Inference
# ---------------------------
MEDGEMMA_PROMPT_TEMPLATE = """
You are an expert clinical co-pilot. Analyze the following patient narrative and any provided images.
Return your analysis ONLY in the following JSON format. Do not include any other text, explanations, or markdown.

{
  "summary": "Your concise clinical summary of the patient's state.",
  "differential": [
    {"dx": "Differential Diagnosis 1", "confidence": 0.8},
    {"dx": "Differential Diagnosis 2", "confidence": 0.15}
  ],
  "recommendations": [
    {"action": "Recommended action 1 (e.g., 'Order chest x-ray')", "risk_level": 1},
    {"action": "Recommended action 2 (e.g., 'Start empiric antibiotics')", "risk_level": 2}
  ],
  "confidence": 0.85
}

Patient Narrative & Context:
{narrative}
"""

def _run_model_inference_sync(inputs: Dict) -> str:
    """Synchronous helper to run model generation in a separate thread."""
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = MODEL.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generation = generation[0][input_len:]
    decoded = PROCESSOR.decode(generation, skip_special_tokens=True)
    return decoded

async def med_gemma_infer(narrative_text: str, image_urls: Optional[List[str]] = None) -> Dict[str, Any]:
    """Performs inference using the loaded local Med-Gemma model."""
    if not MODEL or not PROCESSOR:
        logger.error("Med-Gemma model not loaded. Returning stub.")
        return {"summary": "Model not loaded.", "recommendations": [], "confidence": 0.0}

    prompt = MEDGEMMA_PROMPT_TEMPLATE.format(narrative=narrative_text)
    content = [{"type": "text", "text": prompt}]

    if image_urls:
        for url in image_urls:
            try:
                image = Image.open(requests.get(url, headers={"User-Agent": "clinical-agent/1.0"}, stream=True).raw)
                content.append({"type": "image", "image": image})
            except Exception as e:
                logger.warning(f"Could not load image from {url}: {e}")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist and clinical assistant."}]},
        {"role": "user", "content": content}
    ]

    try:
        inputs = PROCESSOR.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(MODEL.device, dtype=torch.bfloat16)

        decoded = await asyncio.to_thread(_run_model_inference_sync, inputs)
        
        if decoded.strip().startswith("```json"):
            decoded = decoded.strip()[7:-3].strip()
        
        return json.loads(decoded)
    except Exception as e:
        logger.exception(f"Error during local Med-Gemma inference: {e}")
        return {"summary": f"Inference error: {e}", "recommendations": [], "confidence": 0.0}

# ---------------------------
# Post-processing & Planner
# ---------------------------
def generate_soap(med_out: Dict[str, Any]) -> str:
    subjective = med_out.get("summary", "")
    assessment = "\n".join([f"- {d.get('dx', 'N/A')} (conf {d.get('confidence', '?')})" for d in med_out.get("differential", [])])
    plan = "\n".join([f"- {r.get('action', 'N/A')} (risk {r.get('risk_level', '?')})" for r in med_out.get("recommendations", [])])
    return f"Subjective:\n{subjective}\n\nAssessment:\n{assessment}\n\nPlan:\n{plan}"

def icd_tag_stub(summary: str) -> List[Dict[str, str]]:
    tags = []
    s = summary.lower()
    if "pneumonia" in s: tags.append({"code": "J18.9", "desc": "Pneumonia, unspecified organism"})
    if "sepsis" in s: tags.append({"code": "A41.9", "desc": "Sepsis, unspecified"})
    return tags

def plan_from_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks = []
    for rec in recommendations:
        risk = rec.get("risk_level", 1)
        auto_execute = risk <= AUTO_EXECUTE_RISK_LEVEL
        tasks.append({
            "task_id": str(uuid.uuid4()), "description": rec.get("action", "N/A"),
            "risk_level": risk, "auto_execute": auto_execute
        })
    return tasks

# ---------------------------
# EHR Integration Snippet (for simulation)
# ---------------------------
def fhir_patch_resource(resource_type: str, resource_id: str, patch_ops: List[Dict[str, Any]], clinician_auth_token: str):
    logger.info(f"SIMULATING FHIR PATCH to {EHR_BASE}/{resource_type}/{resource_id}")
    return {"status": "simulated_ok", "resourceType": resource_type, "id": resource_id}

# ---------------------------
# Agent Implementation
# ---------------------------
class ClinicalCoPilotAgent:
    def __init__(self, kb: SimpleKnowledgeBase):
        self.db = DB
        self.kb = kb

    async def analyze_patient(self, patient_id: str, fhir_bundle: Dict[str, Any], image_urls: Optional[List[str]] = None) -> str:
        narrative = fhir_to_narrative(fhir_bundle)
        analysis_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        analysis = {"id": analysis_id, "patient_id": patient_id, "narrative": narrative, "status": "queued", "created_at": now, "updated_at": now}
        write_analysis(analysis)
        log_audit(analysis_id, "ingest", {"patient_id": patient_id})
        asyncio.create_task(self._run_analysis(analysis_id, narrative, image_urls))
        return analysis_id

    async def _run_analysis(self, analysis_id: str, narrative: str, image_urls: Optional[List[str]] = None):
        log_audit(analysis_id, "analysis_started", {})
        
        retrieved_docs = rag_retrieve(self.kb, narrative)
        log_audit(analysis_id, "rag_retrieved", {"count": len(retrieved_docs)})

        # Augment narrative with RAG results
        context = "\n".join([d['text'] for d in retrieved_docs])
        augmented_narrative = f"{narrative}\n\nRelevant Clinical Guidelines:\n{context}"

        med_out = await med_gemma_infer(augmented_narrative, image_urls=image_urls)
        log_audit(analysis_id, "medgemma_returned", {"confidence": med_out.get("confidence")})

        soap = generate_soap(med_out)
        icd_tags = icd_tag_stub(med_out.get("summary", ""))
        update_analysis(analysis_id, {
            "medgemma_output": json.dumps(med_out), "soap_note": soap,
            "icd_tags": json.dumps(icd_tags), "status": "ready_for_review"
        })
        log_audit(analysis_id, "analysis_ready", {"icd_tags": icd_tags})

        tasks = plan_from_recommendations(med_out.get("recommendations", []))
        patient_id = self.get_analysis(analysis_id)["patient_id"]
        await self._execute_tasks(analysis_id, patient_id, tasks)

    async def _execute_tasks(self, analysis_id: str, patient_id: str, tasks: List[Dict[str, Any]]):
        for t in tasks:
            log_audit(analysis_id, "task_planned", t)
            if t["auto_execute"]:
                result = await self._perform_action(analysis_id, patient_id, t)
                log_audit(analysis_id, "task_executed", {"task_id": t["task_id"], "result": result})
            else:
                log_audit(analysis_id, "task_requires_approval", t)
                approved, clinician_token = await self._simulate_human_approval(analysis_id, t)
                if approved:
                    result = await self._perform_action(analysis_id, patient_id, t, approved_by="simulated_clinician", clinician_token=clinician_token)
                    log_audit(analysis_id, "task_executed_after_approval", {"task_id": t["task_id"], "result": result})
        update_analysis(analysis_id, {"status": "completed"})
        log_audit(analysis_id, "analysis_completed", {})

    async def _perform_action(self, analysis_id: str, patient_id: str, task: Dict[str, Any], approved_by: Optional[str] = "auto_agent", clinician_token: Optional[str] = EHR_TOKEN) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        if "condition" in task.get("description", "").lower():
            patch = [{"op": "add", "path": "/-", "value": {"code": {"text": task['description']}}}]
            fhir_patch_resource("Patient", patient_id, patch, clinician_token)
        return {"status": "ok", "performed_at": datetime.datetime.utcnow().isoformat(), "task_id": task["task_id"], "approved_by": approved_by}

    async def _simulate_human_approval(self, analysis_id: str, task: Dict[str, Any]) -> tuple[bool, str]:
        logger.info(f"TASK REQUIRES APPROVAL: {task['description']} (Risk: {task['risk_level']})")
        await asyncio.sleep(1.0)
        clinician_jwt = f"fake_jwt_for_clinician_{uuid.uuid4()}"
        logger.info("APPROVAL SIMULATED")
        return True, clinician_jwt

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        cur = self.db.cursor()
        cur.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cur.fetchone()
        if not row: return None
        keys = [description[0] for description in cur.description]
        analysis = dict(zip(keys, row))
        analysis['medgemma_output'] = json.loads(analysis['medgemma_output']) if analysis['medgemma_output'] else None
        analysis['icd_tags'] = json.loads(analysis['icd_tags']) if analysis['icd_tags'] else None
        return analysis

# ---------------------------
# Demo / CLI
# ---------------------------
async def demo_run():
    """Sets up the agent with a local model and runs a demo analysis."""
    load_medgemma_model()
    if not MODEL:
        logger.error("Model could not be loaded. Aborting demo.")
        return

    kb = SimpleKnowledgeBase()
    kb.add_doc("g1", "Guideline: For suspected community-acquired pneumonia in adults, obtain chest x-ray and consider empiric antibiotics when indicated.")
    kb.add_doc("g2", "Guideline: Sepsis screening - qSOFA criteria and lactate measurement recommended.")
    logger.info("Simple Knowledge Base is ready.")

    agent = ClinicalCoPilotAgent(kb=kb)
    fhir_bundle = {
        "entry": [
            {"resource": {"resourceType": "Patient", "name": [{"text": "John Doe"}], "gender": "male", "birthDate": "1953-05-01"}},
            {"resource": {"resourceType": "Observation", "code": {"text": "Temperature"}, "valueQuantity": {"value": 38.5}}},
            {"resource": {"resourceType": "Observation", "code": {"text": "Heart rate"}, "valueQuantity": {"value": 110}}},
            {"resource": {"resourceType": "Condition", "code": {"text": "Cough"}}}
        ]
    }
    
    image_url = "[https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png](https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png)"
    
    logger.info("Submitting patient for analysis with local Med-Gemma model and image...")
    analysis_id = await agent.analyze_patient("P-0001", fhir_bundle, image_urls=[image_url])
    print(f"Analysis queued: {analysis_id}")

    for _ in range(40): # Give more time for local inference
        await asyncio.sleep(1.0)
        rec = agent.get_analysis(analysis_id)
        if rec and rec["status"] in ["completed", "ready_for_review"]:
            print("\n" + "="*25 + " Analysis Completed " + "="*25)
            print(f"SOAP Note:\n{rec['soap_note']}")
            break
        else:
            print(f"Waiting... status={rec['status'] if rec else 'missing'}")

if __name__ == "__main__":
    logger.info("Clinical Co-pilot Agent (Local Model) demo starting...")
    asyncio.run(demo_run())

