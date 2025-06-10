#!/usr/bin/env python3
"""
SarahMemoryAdvCU.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description:
  
  - Tightened question logic to prioritize semantic evaluation and re-routes math queries explicitly
  - Upgraded fallback logic for unknowns to context-aware default based on memory + tone
"""


import re
import logging
import os
import sqlite3
from datetime import datetime
import SarahMemoryGlobals as config
import random
import json
from sentence_transformers import SentenceTransformer, util
import torch
from SarahMemoryGlobals import DATASETS_DIR, MODEL_CONFIG

logger = logging.getLogger("SarahMemoryAdvCU")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(handler)


def log_advcu_event(event, details):
    try:
        db_path = os.path.join(config.DATASETS_DIR, "functions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
CREATE TABLE IF NOT EXISTS advcu_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    event TEXT,
    details TEXT
)
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO advcu_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details))
        conn.commit()
        conn.close()
        logger.info("Logged AdvCU event to functions.db successfully.")
    except Exception as e:
        logger.error(f"Error logging AdvCU event to functions.db: {e}")

INTENT_CACHE_PATH = os.path.join(config.DATASETS_DIR, "intent_override_cache.json")

# === Load Active Sentence Embedding Models ===
intent_descriptions = {
    "greeting": "The user is greeting or starting a conversation.",
    "farewell": "The user is ending the conversation or saying goodbye.",
    "identity": "The user is asking who the AI is or its name.",
    "question": "The user is asking for information, explanation, or facts.",
    "command": "The user is giving a task to perform, like opening or starting something."
}
phi_model = None
minilm_model = None
intent_embeddings_phi = {}
intent_embeddings_mini = {}

for model_name, is_enabled in MODEL_CONFIG.items():
    if not is_enabled:
        continue
    try:
        if "phi" in model_name.lower():
            phi_model = SentenceTransformer(model_name)
            for intent, desc in intent_descriptions.items():
                intent_embeddings_phi[intent] = phi_model.encode(desc, convert_to_tensor=True)
            logger.info(f"[VECTOR INIT] Loaded phi-compatible model: {model_name}")

        elif "mini" in model_name.lower():
            minilm_model = SentenceTransformer(model_name)
            for intent, desc in intent_descriptions.items():
                intent_embeddings_mini[intent] = minilm_model.encode(desc, convert_to_tensor=True)
            logger.info(f"[VECTOR INIT] Loaded MiniLM-compatible model: {model_name}")

    except Exception as e:
        logger.warning(f"[VECTOR INIT FAIL] Model {model_name} skipped: {e}")
def simulated_transformer_intent(text):
    from SarahMemoryGlobals import MODEL_CONFIG
    import logging

    fallback_intent = "statement"

    # Mapping of known model names (from MODEL_CONFIG) to their ideal use cases
    MODEL_PURPOSES = {
        "gpt_4": ["imagine", "empathize", "respond"],
        "gpt_3_5": ["script", "response", "debug"],
        "mistral": ["decode", "summarize", "policy"],
        "llama": ["legal", "sentence", "classify"],
        "gemma": ["calculate", "mathematical", "numeric"],
        "phi": ["function", "technical", "execute"],
        "openchat": ["emotional", "react", "sentiment"],
        "codellama": ["code", "analyze", "script"]
    }

    # Intent types we're mapping across models
    CATEGORIES = ["calculate", "function", "legal", "empathize", "code", "script", "imagine"]

    # Attempt to extract intent from each applicable model
    def get_intent_from_model(model_name, content):
        try:
            from transformers import pipeline
            pipe = pipeline("text-classification", model=model_name)
            result = pipe(content)
            return result[0]['label'].lower()
        except Exception as e:
            logging.warning(f"[Router] Model {model_name} failed: {e}")
            return None

    # Loop through category priorities and route to enabled model
    for category in CATEGORIES:
        for model_key, enabled in MODEL_CONFIG.items():
            if not enabled:
                continue  # skip models that are off
            if category in MODEL_PURPOSES.get(model_key, []):
                intent = get_intent_from_model(model_key, text)
                if intent:
                    return intent

    # Fallback keyword-based basic simulation if no model route succeeded
    probabilities = {
        "greeting": 0.0,
        "farewell": 0.0,
        "question": 0.0,
        "command": 0.0,
        "statement": 0.0
    }
    if "hello" in text or "hi" in text:
        probabilities["greeting"] = 0.8
    if "bye" in text or "goodbye" in text:
        probabilities["farewell"] = 0.8
    if "?" in text:
        probabilities["question"] = 0.7
    if "open" in text or "run" in text:
        probabilities["command"] = 0.7
    probabilities["statement"] = 0.5

    best_intent = max(probabilities, key=probabilities.get)
    logging.debug(f"[Fallback] Simulated transformer intent: {best_intent} with distribution {probabilities}")
    return best_intent


def load_intent_override_cache():
    if os.path.exists(INTENT_CACHE_PATH):
        with open(INTENT_CACHE_PATH, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def classify_intent(text):
    math_symbols = set("+-*/^=%")
    identity_phrases = ["who is", "what is your", "what is their", "who are you", "who am i", "what's your name", "who am i talking to"]
    identity_pattern = re.compile(r"(" + "|".join(re.escape(p) for p in identity_phrases) + r")", re.IGNORECASE)
    math_symbols = set("+-*/^=%")
    if isinstance(text, dict):
        text = text.get("text", "")
    if not isinstance(text, str):
        return "unknown"

    text = text.lower().strip()

    # Check override intent cache
    override_cache = load_intent_override_cache()
    if text in override_cache:
        override_intent = override_cache[text]
        logger.info(f"[OVERRIDE] Intent forced to '{override_intent}' from override cache.")
        log_advcu_event("Intent Override", f"Forced intent: '{override_intent}' for input: '{text}'")
        return override_intent

    # Lexicon triggers
    if any(char in text for char in math_symbols) and not identity_pattern.search(text):
        logger.info("Intent: math | Detected via symbol or pattern match")
        return "question"

    greeting_keywords = r"\b(hello|hi|hey|good morning|good afternoon|greetings|yo|sup|what's up|howdy|hiya|how are you|hey there|what's crackin|what's happening|whats good|yo yo|morning sunshine|evening chief)\b"
    farewell_keywords = r"\b(bye|goodbye|see you|farewell|later|cya|exit|peace out|i'm leaving|talk to you later|i'm out|laters|catch you later|deuces)\b"
    question_keywords = r"\b(what|how|why|where|when|who|can you|could you|do you|will you|is it|does it|would you|did you|should i|am i|are you|may i|shall i|have you|how come|what if|is there|how much|how many|which one)\b"
    command_keywords = r"\b(open|play|start|launch|do|run|execute|begin|initiate|turn on|trigger|access|enable|boot|show|display|scan|search|lookup|analyze|load|install|activate|fire up|kick off|bring up|pop open|turn up|pull up|make it go|let's roll|spin up|get going|deploy|terminate|end|close|quit|turn down|turn on|turn off|fire the engine)\b"

    identity_keywords = [
        "your name", "who are you", "what's your name", "tell me your name",
        "do you have a name", "identify yourself", "can you tell me your name",
        "are you sarah", "which ai are you", "are you human", "who am i talking to"
    ]

    question_phrases = [
        "what is", "who is", "how do", "how to", "where is", "what are", "why is", "when is",
        "how many", "what's", "how come", "how can", "how may I", "when may I", "when will", "when can I"
        "can you explain", "could you tell me", "tell me what", "tell me when", "tell me who", "tell me why", "tell me how",
        "would you mind explaining", "is there a way to", "can i find", "is it true that", "when are", "why can't I", "when can", "what can" 
    ]

    if any(q in text for q in question_phrases):
        return "question"
    if any(g in text for g in greeting_keywords.split('|')):
        return "greeting"
    if any(i in text for i in identity_keywords):
        return "identity"
    if any(c in text for c in command_keywords.split('|')):
        return "command"

    try:
        if re.search(greeting_keywords, text, re.IGNORECASE):
            intent = "greeting"
        elif re.search(farewell_keywords, text, re.IGNORECASE):
            intent = "farewell"
        elif re.search(question_keywords, text, re.IGNORECASE):
            intent = "question"
        elif re.search(command_keywords, text, re.IGNORECASE):
            intent = "command"
        else:
            intent = simulated_transformer_intent(text)
            db_path = os.path.join(config.DATASETS_DIR, "functions.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS unclassified_intents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                raw_input TEXT,
                assumed_intent TEXT
            )
            """)
            timestamp = datetime.now().isoformat()
            cursor.execute("INSERT INTO unclassified_intents (timestamp, raw_input, assumed_intent) VALUES (?, ?, ?)",
                           (timestamp, text, intent))
            conn.commit()
            conn.close()
                # Vector-based semantic scoring
        if getattr(config, "LEARNING_PHASE_ACTIVE", False):  # Add this guard in SarahMemoryGlobals
            query_phi = phi_model.encode(text, convert_to_tensor=True)
            query_mini = minilm_model.encode(text, convert_to_tensor=True)

        phi_scores = {k: float(util.pytorch_cos_sim(query_phi, emb)[0][0]) for k, emb in intent_embeddings_phi.items()}
        mini_scores = {k: float(util.pytorch_cos_sim(query_mini, emb)[0][0]) for k, emb in intent_embeddings_mini.items()}

        combined_scores = {k: round((phi_scores[k] + mini_scores[k]) / 2, 3) for k in intent_descriptions.keys()}
        best_scoring_intent = max(combined_scores, key=combined_scores.get)
        logger.info(f"[INTENT AI] Vector Scores → {combined_scores} | Final: {best_scoring_intent}")
        intent = best_scoring_intent
        log_advcu_event("Classify Intent", f"Input: '{text}' classified as '{intent}'")
        return intent
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        log_advcu_event("Classify Intent Error", f"Error for text '{text}': {e}")
        return "statement"
    
def evaluate_similarity(text1, text2):
    """Simple semantic similarity metric using shared words."""
    try:
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if union else 0.0
    except Exception as e:
        logger.error(f"Similarity evaluation error: {e}")
        return 0.0

def get_vector_score(text1, text2):
    """Vector score using simple numeric encoding for quick approximation."""
    try:
        vec1 = [ord(c) for c in text1 if c.isalnum()]
        vec2 = [ord(c) for c in text2 if c.isalnum()]
        if not vec1 or not vec2:
            return 0.0
        score = sum(min(a, b) for a, b in zip(vec1, vec2)) / max(sum(vec1), sum(vec2))
        return min(score, 1.0)
    except Exception as e:
        logger.error(f"Vector score error: {e}")
        return 0.0

def parse_command(text):
    text = text.lower().strip()
    tokens = text.split()
    command_synonyms = ["open", "start", "launch", "run", "execute", "fire up", "boot", "initiate", "enable", "terminate", "kill", "quit", "end", "close", "maximize", "minimize", "turn on", "turn off", "turn down", "turn up"]
    for verb in command_synonyms:
        if verb in tokens:
            index = tokens.index(verb)
            target = ' '.join(tokens[index + 1:]).strip()
            if target:
                return {"action": verb, "target": target}
    return None

def classify_intent_with_confidence(text):
    try:
        if 'phi_model' not in globals() or 'minilm_model' not in globals():
            logger.warning("[CONFIDENCE] Required models are not loaded in memory.")
            return "statement", 0.0

        if 'intent_embeddings_phi' not in globals() or 'intent_embeddings_mini' not in globals():
            logger.warning("[CONFIDENCE] Intent embeddings missing or corrupted.")
            return "statement", 0.0

        query_phi = phi_model.encode(text, convert_to_tensor=True)
        query_mini = minilm_model.encode(text, convert_to_tensor=True)
        phi_scores = {k: float(util.pytorch_cos_sim(query_phi, emb)[0][0]) for k, emb in intent_embeddings_phi.items()}
        mini_scores = {k: float(util.pytorch_cos_sim(query_mini, emb)[0][0]) for k, emb in intent_embeddings_mini.items()}
        combined_scores = {k: round((phi_scores[k] + mini_scores[k]) / 2, 3) for k in intent_descriptions.keys()}
        best_intent = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[best_intent]
        logger.info(f"[CONFIDENCE SCORING] {combined_scores}")
        return best_intent, confidence
    except Exception as e:
        logger.error(f"[CONFIDENCE SCORING ERROR] {e}")
        return "statement", 0.0

def split_intent_chain(text):
    return [part.strip() for part in re.split(r'[.;]', text) if part.strip()]
if __name__ == "__main__":
    test_inputs = [
        "Hey Sarah, how are you?",
        "Can you open my email?",
        "Goodbye for now",
        "What is your name?",
        "What is 5 plus 5?",
        "Calculate the area of a square",
        "I really enjoy using this AI platform."
    ]
    for phrase in test_phrases:
        parts = split_intent_chain(phrase)
        for part in parts:
            intent, conf = classify_intent_with_confidence(part)
            print(f"Input: {part} → Intent: {intent}, Confidence: {conf:.2f}")
        sim_score = evaluate_similarity(phrase, "Hello Sarah, what is your name?")
        print(f"Similarity to 'Hello Sarah, what is your name?': {sim_score:.2f}")
        cmd = parse_command(phrase)
        if cmd:
            print(f"Command parsed: {cmd}")
