#!/usr/bin/env python3
"""
SarahMemoryCompare.py <Version 7.1 Enhanced>
✅ Primary Function
Compare multiple sources (Local, Web, API variants)
Score responses based on:
- Semantic similarity
- Token overlap
- Embedding vector score
- Emotional richness (optional)
- Source priority
Logs to audit files and enables auto-correction and supervised voting
"""

import logging
import datetime
import json
import os
from sentence_transformers import SentenceTransformer
import SarahMemoryGlobals as config
from SarahMemoryGlobals import (
    API_RESPONSE_CHECK_TRAINER, COMPARE_VOTE, DATASETS_DIR, DEBUG_MODE,
    MULTI_MODEL, MODEL_CONFIG, LOCAL_DATA_ENABLED, WEB_RESEARCH_ENABLED, API_RESEARCH_ENABLED
)
from SarahMemoryWebSYM import WebSemanticSynthesizer
from SarahMemoryDatabase import record_qa_feedback, auto_correct_dataset_entry, tokenize_text
from SarahMemoryAdvCU import evaluate_similarity, get_vector_score
from SarahMemoryAPI import send_to_api
from SarahMemoryResearch import APIResearch

logger = logging.getLogger('SarahMemoryCompare')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

def get_active_sentence_model():
    if MULTI_MODEL:
        for model_name, enabled in MODEL_CONFIG.items():
            if enabled:
                try:
                    return SentenceTransformer(model_name)
                except Exception as e:
                    logger.warning(f"Model load failed: {model_name} → {e}")
    return SentenceTransformer('all-MiniLM-L6-v2')  # fallback

def fetch_all_sources(user_text, intent):
    sources = {}

    if LOCAL_DATA_ENABLED:
        from SarahMemoryPersonality import get_reply_from_db
        try:
            local_resp = get_reply_from_db(intent)
            if local_resp:
                sources['local'] = local_resp if isinstance(local_resp, str) else local_resp[0]
        except Exception as e:
            logger.warning(f"[COMPARE] Local source failed: {e}")

    if WEB_RESEARCH_ENABLED:
        try:
            web_resp = WebSemanticSynthesizer.synthesize_response("", user_text)
            if web_resp:
                sources['web'] = web_resp
        except Exception as e:
            logger.warning(f"[COMPARE] Web source failed: {e}")

    if API_RESEARCH_ENABLED:
        try:
            result = APIResearch.query(user_text, intent)
            if result and result.get("data"):
                sources[result.get("source", "api")] = result.get("data")
        except Exception as e:
            logger.warning(f"[COMPARE] API source failed: {e}")

    return sources

def compare_reply(user_text, generated_response, intent="general"):
    if not API_RESPONSE_CHECK_TRAINER:
        logger.info("Comparison skipped: API_RESPONSE_CHECK_TRAINER is False")
        return {"status": "SKIPPED", "feedback": "API response comparison is disabled."}

    try:
        response_pool = fetch_all_sources(user_text, intent)
        if not response_pool:
            return {"status": "ERROR", "feedback": "No sources available for comparison."}

        if isinstance(generated_response, list):
            generated_response = " ".join(generated_response)

        best_score = 0.0
        best_match = None

        for source_name, response in response_pool.items():
            if isinstance(response, list):
                response = " ".join(response)

            local_tokens = tokenize_text(generated_response)
            alt_tokens = tokenize_text(response)
            token_overlap = len(set(local_tokens).intersection(set(alt_tokens)))
            total_tokens = max(len(set(local_tokens + alt_tokens)), 1)
            overlap_ratio = token_overlap / total_tokens

            vector_score = get_vector_score(generated_response, response)
            similarity_score = evaluate_similarity(generated_response, response)
            weighted_confidence = round((similarity_score * 0.5) + (vector_score * 0.3) + (overlap_ratio * 0.2), 3)

            if weighted_confidence > best_score:
                best_score = weighted_confidence
                best_match = {
                    "source": source_name,
                    "response": response,
                    "confidence": weighted_confidence,
                    "similarity_score": round(similarity_score, 3),
                    "vector_score": round(vector_score, 3),
                    "token_overlap": round(overlap_ratio, 3)
                }

        feedback = "HIT" if best_score >= 0.78 else "MISS"
        timestamp = datetime.datetime.now().isoformat()

        record_qa_feedback(
            user_text,
            score=1 if feedback == "HIT" else 0,
            feedback=f"{feedback} | confidence={best_score} | timestamp={timestamp}"
        )

        audit_path = os.path.join(config.DATASETS_DIR, "logs", f"compare_audit_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        with open(audit_path, 'w') as f:
            json.dump({
                "user_text": user_text,
                "response_given": generated_response,
                "best_match": best_match,
                "all_sources": response_pool,
                "status": feedback,
                "timestamp": timestamp
            }, f, indent=2)

        if COMPARE_VOTE:
            logger.info("[COMPARE_VOTE] Prompt user: Was this a helpful response? [Yes/No]")

        return {
            "status": feedback,
            "confidence": best_score,
            "matched_source": best_match.get("source") if best_match else None,
            "similarity_score": best_match.get("similarity_score") if best_match else 0.0,
            "vector_score": best_match.get("vector_score") if best_match else 0.0,
            "token_overlap": best_match.get("token_overlap") if best_match else 0.0,
            "api_response": best_match.get("response") if best_match else ""
        }

    except Exception as e:
        logger.error(f"[COMPARE ERROR] {e}")
        return {"status": "ERROR", "feedback": str(e)}


if __name__ == "__main__":
    input_text = input("Enter prompt for test comparison: ")
    test_response = input("Enter AI-generated response: ")
    intent = input("Enter intent (default: general): ") or "general"
    result = compare_reply(input_text, test_response, intent=intent)
    print("\nTest Result:\n", json.dumps(result, indent=2))
