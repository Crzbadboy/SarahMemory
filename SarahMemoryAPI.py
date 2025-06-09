#!/usr/bin/env python3
# ============================================
# SarahMemoryAPI.py <Version 7.0 Unified> By Brian Lee Baros
# Description: Unified API Connector for Class 3 Research
# Supports: OpenAI, Claude, Mistral, Gemini, HuggingFace
# ============================================

import json
import logging
import os
import requests
import sqlite3
from datetime import datetime

import SarahMemoryGlobals as config
from SarahMemoryGlobals import DATASETS_DIR, run_async
from SarahMemoryAdvCU import classify_intent
from SarahMemoryAiFunctions import get_context
from SarahMemoryAdaptive import simulate_emotion_response

logger = logging.getLogger("SarahMemoryAPI")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

API_DISABLED = not config.API_RESEARCH_ENABLED
# API KEYS AND ENDPOINTS (Configure per provider)
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "huggingface": os.getenv("HF_API_KEY")
}

API_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "claude": "https://api.anthropic.com/v1/complete",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "huggingface": "https://api-inference.huggingface.co/models/{model}"
}

DEFAULT_MODELS = {
    "openai": "gpt-4",
    "claude": "claude-v1.3",
    "mistral": "mistral-medium",
    "gemini": "gemini-pro",
    "huggingface": "bigscience/bloom"
}

cache = {}

ROLE_MAP = {
    "question": "expert researcher",
    "command": "AI task executor",
    "debug": "senior software engineer",
    "explanation": "technical expert",
    "teaching": "university professor",
    "emotion": "empathetic therapist",
    "identity": "AI assistant with personality",
    "joke": "stand-up comedian",
    "story": "creative storyteller",
    "finance": "financial analyst",
    "math": "mathematician",
    "science": "scientist",
    "medical": "licensed medical expert",
    "emergency": "crisis advisor",
    "unknown": "general-purpose AI assistant"
}

def log_api_event(event, details):
    try:
        db_path = os.path.abspath(os.path.join(config.DATASETS_DIR, "system_logs.db"))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_integration_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT
            )
        """)
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO api_integration_events (timestamp, event, details) VALUES (?, ?, ?)",
                       (timestamp, event, details[:500]))
        conn.commit()
        conn.close()
        logger.info(f"[LOG] {event}: {details[:100]}...")
    except Exception as e:
        logger.error(f"[API Log Error] {e}")

def fallback_provider(current):
    priority = ["openai", "claude", "mistral", "gemini", "huggingface"]
    return next((p for p in priority if p != current and getattr(config, f"{p.upper()}_API", False)), None)

def build_advanced_prompt(user_input, intent="question", tone="friendly", complexity="adult", provider="unknown"):
    import logging
    research_path_logger = logging.getLogger("ResearchPathLogger")

    role = ROLE_MAP.get(intent, "general-purpose AI assistant")
    context_snippets = get_context()
    recent_context = "\n".join([c['input'] for c in context_snippets[-3:]]).strip()
    emotion = simulate_emotion_response()
    mood = f"Joy: {emotion.get('joy',0):.2f}, Trust: {emotion.get('trust',0):.2f}, Fear: {emotion.get('fear',0):.2f}, Anger: {emotion.get('anger',0):.2f}, Surprise: {emotion.get('surprise',0):.2f}"

    context_line = f"CONTEXT: {recent_context}" if recent_context else "CONTEXT: None (no recent conversation history)"

    prompt = f"""
ROLE: You are a {role}.
INTENT: {intent.upper()}
TONE: {tone}
COMPLEXITY: {complexity}
MOOD PROFILE: {mood}
{context_line}
QUERY: {user_input}

Respond clearly, helpfully, and concisely based on the current emotional tone.
""".strip()

    research_path_logger.info(f"API Call → Provider: {provider}, emotion: {emotion}")
    return prompt, emotion

def send_to_api(user_input, provider="openai", intent="question", tone="friendly", complexity="adult", model=None):

    import logging
    research_path_logger = logging.getLogger("ResearchPathLogger")
   
    if  not config.API_RESEARCH_ENABLED:
        logger.warning("[BLOCKED] API research disabled in Globals.")
        return {"source": provider, "data": None, "prompt": None, "intent": "n/a"}

    key = API_KEYS.get(provider)
    if not key:
        return {"source": provider, "data": "API key missing."}

    model = model or DEFAULT_MODELS.get(provider, "gpt-4")
    prompt, emotion = build_advanced_prompt(user_input, intent, tone, complexity)
    if prompt in cache:
        return {"source": provider, "data": cache[prompt]}

    headers, payload, url = {}, {}, API_URLS.get(provider)

    if provider == "openai":
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 800
        }
    elif provider == "claude":
        headers = {"x-api-key": key, "Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens_to_sample": 800,
            "stop_sequences": ["\n\n"]
        }
    elif provider == "mistral":
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
    elif provider == "gemini":
        url += f"?key={key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
    elif provider == "huggingface":
        url = url.format(model=model)
        headers = {"Authorization": f"Bearer {key}"}
        payload = {"inputs": prompt}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=config.API_TIMEOUT)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        data = response.json()
         # [PATCH] Safely extract content for OpenAI - handles message or text
        if provider == "openai":
            if "message" in data.get("choices", [{}])[0]:  # [PATCH]
                content = data.get("choices", [{}])[0]["message"].get("content", "").strip()
            else:  # [PATCH]
                content = data.get("choices", [{}])[0].get("text", "").strip()
        elif provider == "claude":
            content = data.get("completion", "").strip()
        elif provider == "gemini":
            content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        elif provider == "huggingface":
            content = data[0].get("generated_text", "") if isinstance(data, list) else str(data)
        else:
            content = str(data)

        
        # ✅ DEBUG RESPONSE LOGGING (must be outside exception)
         # [PATCH] Log extracted and raw data for OpenAI debug
        logger.warning(f"[OPENAI DEBUG RESPONSE] Raw data: {data}")  # [PATCH]
        logger.warning(f"[OPENAI DEBUG RESPONSE] Parsed content: {content}")  # [PATCH]



        if not content:
            raise ValueError("No usable content returned.")

        cache[prompt] = content
        log_api_event(f"{provider.upper()} API Success", f"Prompt: {prompt[:80]} | Response: {content[:80]}")

        return {
            "source": provider,
            "data": content,
            "prompt": prompt,
            "intent": intent,
            "tone": tone,
            "emotion": emotion
        }

    except Exception as e:
        logger.error(f"[{provider.upper()} API Exception] {e}")
        if not config.API_RESEARCH_ENABLED:
            return {"source": provider, "data": None, "intent": "n/a"}
        
        fallback = fallback_provider(provider)
        if fallback:
            logger.warning(f"[Fallback Triggered] Switching from {provider} to {fallback}")
            return send_to_api(user_input, provider=fallback, intent=intent, tone=tone, complexity=complexity)
        research_path_logger.info(f"SarahMemoryAPI.py -> def send_to_api, User Input: {user_input}, API Call →  Source: {provider}, Intent: {intent}, Tone: {tone}, Complexity: {complexity}")
        return {"source": provider, "data": "API request failed."}

def send_to_api_async(user_input, provider, callback):
        
    def async_task():
        result = send_to_api(user_input, provider)
        callback(result)
    run_async(async_task)
    
def run_cognitive_analysis(text, provider="openai"):
    """
    Unified cognitive classification (emotion, tone, sentiment) using supported API providers.
    """
    try:
        if provider == "openai":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a cognitive and emotional classifier. Return tone, sentiment and emotion from the user's message."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=300
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"[OpenAI Cognitive] {result}")
            log_cognitive_event("OpenAI Text Analysis", result)
            return {"source": "openai", "result": result}

        elif provider == "microsoft":
            # Stubbed Microsoft version (if API available)
            return {"error": "Microsoft Cognitive Services not configured."}

        elif provider == "claude":
            return {"error": "Claude cognitive analysis not yet implemented."}

        elif provider == "gemini":
            return {"error": "Gemini cognitive analysis not yet implemented."}

        elif provider == "huggingface":
            return {"error": "HuggingFace cognitive analysis not yet implemented."}

        elif provider == "mistral":
            return {"error": "Mistral cognitive analysis not yet implemented."}

        else:
            return {"error": f"Unknown provider '{provider}'"}

    except Exception as e:
        logger.error(f"[Cognitive Error:{provider}] {e}")
        return {"error": str(e)}

def analyze_image(image_path):
    logger.warning("Image analysis via OpenAI or LMMs not yet supported.")
    return {"error": "Image analysis not implemented."}

if __name__ == "__main__":
    logger.info("[TEST] Unified API Prompt Framework v7.0")
    result = send_to_api("Explain how a fusion reactor works.", provider="openai")
    print(json.dumps(result, indent=2))
