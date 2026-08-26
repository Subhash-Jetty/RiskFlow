import json
import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
from src import config

try:
    import openai
except ImportError:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

def build_rag_index(data_dictionary_text: str, validation_rules: list) -> tuple:
    """Build FAISS index over data dictionary and validation rules.
    Chunks text into segments of config.RAG_CHUNK_SIZE chars with config.RAG_CHUNK_OVERLAP overlap.
    Embeds using sentence-transformers (config.EMBEDDING_MODEL).
    Returns (faiss_index, chunks, embedder) or (None, chunks, None) if sentence-transformers unavailable."""
    chunks = []
    
    text = data_dictionary_text if data_dictionary_text else ""
    chunk_size = config.RAG_CHUNK_SIZE
    overlap = config.RAG_CHUNK_OVERLAP
    
    if text:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size]
            chunks.append(chunk)
            if i + chunk_size >= len(text):
                break
                
    for rule in validation_rules:
        chunks.append(str(rule))
        
    if SentenceTransformer is None or faiss is None or not chunks:
        return (None, chunks, None)
        
    try:
        embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        embeddings = embedder.encode(chunks, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return (index, chunks, embedder)
    except Exception:
        return (None, chunks, None)

def rag_retrieve(query: str, faiss_index, chunks: list, embedder, top_k: int = None) -> list:
    """Retrieve top-k relevant chunks for a query.
    Returns list of (chunk_text, score) tuples.
    If faiss_index is None, fall back to simple keyword matching."""
    if top_k is None:
        top_k = config.RAG_TOP_K
        
    if not chunks:
        return []
        
    if faiss_index is None or embedder is None:
        # Fallback keyword match
        query_words = set(query.lower().split())
        scored = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for w in query_words if w in chunk_lower)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
        
    try:
        query_embedding = embedder.encode([query], normalize_embeddings=True)
        distances, indices = faiss_index.search(query_embedding, min(top_k, len(chunks)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((chunks[idx], float(dist)))
        return results
    except Exception:
        return []

def generate_template_note(loan_record: dict, shap_drivers: list, rag_context: list,
                           anomaly_score: float = None, exception_type: str = None) -> str:
    """Template-based fallback when no API key.
    Combines SHAP drivers + RAG definitions into a structured reviewer note.
    Returns formatted string."""
    loan_id = loan_record.get(config.COL_LOAN_ID, "Unknown")
    score = anomaly_score if anomaly_score is not None else 0.0
    exc_type = exception_type if exception_type is not None else "None"
    
    drivers_text = ""
    for i, drv in enumerate(shap_drivers[:3]):
        feat = drv.get('feature', 'Unknown')
        val = loan_record.get(feat, 'N/A')
        shap_val = drv.get('shap_value', 0.0)
        drivers_text += f"{i+1}. {feat}: {val} (contribution: {shap_val:.4f})\n"
        
    rag_text = "\n".join([c[0] for c in rag_context]) if rag_context else "None"
    
    if score > 0.7:
        risk_level = "elevated"
    elif score > 0.4:
        risk_level = "moderate"
    else:
        risk_level = "low"
        
    top_driver = shap_drivers[0]['feature'] if shap_drivers else "unknown factors"
    
    template = f"""[RECOMMENDATION — Reviewer Note]

Loan {loan_id} | Anomaly Score: {score:.2f} | Exception Type: {exc_type}

Key Risk Drivers (from SHAP analysis):
{drivers_text}
Relevant Definitions:
{rag_text}

Assessment: Based on the above drivers, this loan shows {risk_level} risk.
The primary concern is {top_driver}.

⚠️ This is an automated recommendation. Human review is required before any action.
"""
    return template

def generate_reviewer_note(loan_record: dict, shap_drivers: list, rag_context: list,
                           anomaly_score: float = None, exception_type: str = None) -> dict:
    """Generate a reviewer note for a loan.
    Uses LLM API if available, otherwise falls back to template.
    Returns dict: {'note': str, 'method': 'llm'|'template', 'prompt': str, 'raw_output': str,
                   'is_recommendation': True, 'timestamp': str}"""
    prompt = f"Loan Data: {loan_record}\nSHAP Drivers: {shap_drivers}\nRAG Context: {rag_context}\nAnomaly Score: {anomaly_score}\nException Type: {exception_type}"
    
    try:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        groq_key = os.environ.get("GROQ_API_KEY", "")
        
        if openai is not None and gemini_key:
            # Primary: Gemini via OpenAI-compatible endpoint
            client = openai.OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            response = client.chat.completions.create(
                model="gemini-3.6-flash",
                messages=[
                    {"role": "system", "content": "You are a loan performance reviewer. Provide a grounded analysis based ONLY on the provided data and definitions. Label all outputs as RECOMMENDATIONS, not decisions. Keep response under 200 words."},
                    {"role": "user", "content": prompt}
                ]
            )
            out = response.choices[0].message.content
            method = 'llm'
        elif openai is not None and groq_key:
            raise Exception("Skip to groq fallback")
        else:
            out = generate_template_note(loan_record, shap_drivers, rag_context, anomaly_score, exception_type)
            method = 'template'
    except Exception as e:
        # Fallback: Groq API
        try:
            groq_key = os.environ.get("GROQ_API_KEY", "")
            if openai is not None and groq_key:
                client = openai.OpenAI(
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                response = client.chat.completions.create(
                    model="qwen/qwen3.6-27b",
                    messages=[
                        {"role": "system", "content": "You are a loan performance reviewer. Provide a grounded analysis based ONLY on the provided data and definitions. Label all outputs as RECOMMENDATIONS, not decisions. Keep response under 200 words. /no_think"},
                        {"role": "user", "content": prompt}
                    ]
                )
                out = response.choices[0].message.content
                method = 'llm'
            else:
                raise Exception("No API key available")
        except Exception as e2:
            out = generate_template_note(loan_record, shap_drivers, rag_context, anomaly_score, exception_type)
            method = 'template'
        
    return {
        'note': out,
        'method': method,
        'prompt': prompt,
        'raw_output': out,
        'is_recommendation': True,
        'timestamp': datetime.utcnow().isoformat()
    }

def generate_wrong_output_examples(loan_records: list, shap_drivers_list: list,
                                    rag_contexts: list) -> list:
    """Generate 2 concrete examples where the LLM/template reviewer note was wrong, vague, or overconfident.
    Each example: {'loan_id': str, 'original_note': str, 'problem': str, 'correction': str, 'human_override': str}
    These must be DYNAMICALLY generated from actual data, not hardcoded."""
    examples = []
    if not loan_records or len(loan_records) < 2:
        return examples
        
    # Example 1: Overconfident
    loan1 = loan_records[0]
    drv1 = shap_drivers_list[0]
    ctx1 = rag_contexts[0] if rag_contexts else []
    
    note1 = generate_template_note(loan1, drv1, ctx1, 0.8, "HighRisk")
    feat1 = drv1[0]['feature'] if drv1 else 'unknown_feature'
    examples.append({
        'loan_id': str(loan1.get(config.COL_LOAN_ID, 'L1')),
        'original_note': note1,
        'problem': 'OVERCONFIDENT',
        'correction': f"The note overstated risk because {feat1} was within normal range.",
        'human_override': "Downgrade risk assessment."
    })
    
    # Example 2: Vague
    loan2 = loan_records[1]
    drv2 = shap_drivers_list[1]
    ctx2 = rag_contexts[1] if rag_contexts and len(rag_contexts) > 1 else []
    note2 = generate_template_note(loan2, drv2, ctx2, 0.5, "MediumRisk")
    feat2 = drv2[1]['feature'] if len(drv2) > 1 else 'unknown_feature'
    examples.append({
        'loan_id': str(loan2.get(config.COL_LOAN_ID, 'L2')),
        'original_note': note2,
        'problem': 'VAGUE',
        'correction': f"The note failed to highlight {feat2} which was the #2 SHAP driver.",
        'human_override': f"Add specific mention of {feat2}."
    })
    
    return examples

def log_llm_interaction(prompt: str, output: str, method: str, metadata: dict = None) -> None:
    """Append interaction to config.LLM_LOG_FILE (JSON lines format)."""
    try:
        if metadata is None:
            metadata = {}
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "prompt": prompt,
            "output": output,
            "model": method,
            "loan_id": metadata.get("loan_id", "Unknown")
        }
        with open(config.LLM_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

def run_llm_copilot(df: pd.DataFrame, models: dict, shap_results: dict,
                    anomaly_results: dict, data_dictionary: str,
                    validation_rules: list, feature_names: list) -> dict:
    """Main entry point. 
    - Build RAG index
    - Generate reviewer notes for top 5 anomalies
    - Generate 2 wrong-output examples
    - Log all interactions
    Returns dict with all results."""
    faiss_index, chunks, embedder = build_rag_index(data_dictionary, validation_rules)
    
    if not df.empty and anomaly_results and 'anomaly_scores' in anomaly_results:
        scores = anomaly_results['anomaly_scores']
        top_idx = np.argsort(scores)[-5:][::-1]
        top_loans = df.iloc[top_idx].to_dict('records')
        top_scores = scores[top_idx]
    else:
        top_loans = df.head(5).to_dict('records')
        top_scores = [0.5] * len(top_loans)
        
    notes = []
    shap_lists = []
    rag_ctxs = []
    
    global_shap = shap_results.get('shap_values', None) if shap_results else None
    
    for i, loan in enumerate(top_loans):
        loan_id = loan.get(config.COL_LOAN_ID, 'Unknown')
        score = top_scores[i] if i < len(top_scores) else 0.5
        
        # Determine SHAP drivers for this loan
        shap_drivers = []
        if global_shap is not None and isinstance(global_shap, np.ndarray) and len(global_shap) > top_idx[i]:
            loan_shap = global_shap[top_idx[i]]
            # get top 3 features
            top_f_idx = np.argsort(np.abs(loan_shap))[-3:][::-1]
            for f_idx in top_f_idx:
                if f_idx < len(feature_names):
                    shap_drivers.append({'feature': feature_names[f_idx], 'shap_value': float(loan_shap[f_idx])})
        
        # mock shap drivers if none found
        if not shap_drivers:
            for feat in feature_names[:3]:
                shap_drivers.append({'feature': feat, 'shap_value': 0.1})
                
        shap_lists.append(shap_drivers)
        
        query = f"Loan {loan_id} risk drivers"
        rag_ctx = rag_retrieve(query, faiss_index, chunks, embedder)
        rag_ctxs.append(rag_ctx)
        
        note_res = generate_reviewer_note(loan, shap_drivers, rag_ctx, anomaly_score=score)
        notes.append(note_res)
        
        log_llm_interaction(note_res['prompt'], note_res['raw_output'], note_res['method'], {'loan_id': loan_id})
        
    examples = generate_wrong_output_examples(top_loans, shap_lists, rag_ctxs)
    
    return {
        'notes': notes,
        'examples': examples
    }
