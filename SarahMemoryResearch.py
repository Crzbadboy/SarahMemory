#!/usr/bin/env python3
# ============================================
# SarahMemoryResearch.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
# Description: Multi-Class Research Engine (Local > Web > API) + Parallelization
# ============================================

import logging
import asyncio
import aiohttp
import json
import os
import re
import time
import hashlib
import html
from bs4 import BeautifulSoup
import winreg

import SarahMemoryGlobals as config
from SarahMemoryAdvCU import classify_intent
from SarahMemoryAiFunctions import get_context, add_to_context
from SarahMemoryAPI import send_to_api
from SarahMemoryDatabase import search_answers, search_responses
from SarahMemoryGlobals import import_other_data, MODEL_CONFIG
from SarahMemoryWebSYM import WebSemanticSynthesizer

# Main System Logger
logger = logging.getLogger("SarahMemoryResearch")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Separate Debug Logger for Query Research Path of each typed query submitted. 
debug_log_path = os.path.join(config.BASE_DIR, "data", "logs", "research.log")
research_debug_logger = logging.getLogger("ResearchDebug")
research_debug_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(debug_log_path, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not research_debug_logger.hasHandlers():
    research_debug_logger.addHandler(file_handler)

# Internal Research Cache
research_cache = {}

# Web Endpoints
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
FREE_DICTIONARY_API = "https://www.thefreedictionary.com/"
OPENLIBRARY_API = "https://openlibrary.org/search.json?q="
DUCKDUCKGO_HTML = "https://duckduckgo.com/html/?q="

# 🧐 Offline Static Fallback Facts
STATIC_FACTS = {
    "what is the speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "what is the boiling point of water": "The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.",
    "what is pi": "Pi is a mathematical constant approximately equal to 3.14159."
}

class LocalResearch:
    @staticmethod
    def search(query, intent):
        try:
            research_debug_logger.debug(f"[SEARCH INIT] Query: {query} | Intent: {intent}")

            cached_answers = search_answers(query)
            if cached_answers:
                research_debug_logger.debug(f"[QA CACHE HIT] Result: {cached_answers[0]}")
                logger.info("[Class 1] Found match in QA Cache.")
                return {"source": "local", "intent": intent, "results": [{"file": "QA Cache", "snippet": cached_answers[0]}]}

            response_answers = search_responses(query)
            if response_answers:
                research_debug_logger.debug(f"[PERSONALITY RESPONSE HIT] Result: {response_answers[0]}")
                logger.info("[Class 1] Found match in Personality Responses.")
                return {"source": "local", "intent": intent, "results": [{"file": "Personality1", "snippet": response_answers[0]}]}

            results = []
            for file, content in import_other_data().items():
                if query.lower() in content.lower():
                    research_debug_logger.debug(f"[IMPORT HIT] File: {file}")
                    results.append({"file": file, "snippet": content[:300].replace("\n", " ")})
            if results:
                logger.info("[Class 1] Found match in imported datasets.")
                return {"source": "local", "intent": intent, "results": results}

            logger.info("[Class 1] No DB hit, trying configured ensemble LLM response synthesis.")
            from sentence_transformers import SentenceTransformer
            enabled_models = [
                (model_key, enabled) for model_key, enabled in MODEL_CONFIG.items() if enabled
            ]

            responses = []
            for model_key, _ in enabled_models:
                try:
                    model_path = os.path.join(config.MODELS_DIR, model_key.replace("/", "_"))
                    if not os.path.exists(model_path):
                        logger.warning(f"[Missing Model] {model_key} not found at expected path: {model_path}")
                        research_debug_logger.debug(f"[MISSING MODEL] {model_key} not found.")
                        continue
                    model = SentenceTransformer(model_path)
                    embedding = model.encode(query)
                    if embedding is None or len(embedding) == 0:
                        raise ValueError("Generated embedding was empty.")
                    research_debug_logger.debug(f"[MODEL SUCCESS] {model_key} returned valid embedding.")
                    responses.append(f"Model [{model_key}] understood the query and processed it.")
                except Exception as e:
                    logger.warning(f"[LLM Failure] {model_key}: {e}")
                    research_debug_logger.debug(f"[MODEL FAILURE] {model_key}: {str(e)}")

            if responses:
                combined = " | ".join(responses)
                return {"source": "ensemble-llm", "intent": intent, "results": [{"file": "Offline LLMs", "snippet": combined}]}

            logger.info("[Class 1] Attempting WebSYM Fallback.")
            synthesized = WebSemanticSynthesizer.synthesize_response("", query)
            if synthesized and "couldn't find reliable information" not in synthesized.lower():
            
                return {"source": "local-websym", "intent": intent, "results": [{"file": "DynamicWebSYM", "snippet": synthesized}]}

            static_fact = STATIC_FACTS.get(query.lower().strip())
            if static_fact:
                logger.info("[Class 1] Static fallback match found.")
                
                return {"source": "static-fallback", "intent": intent, "results": [{"file": "Hardcoded", "snippet": static_fact}]}

            logger.warning("[Class 1] No local match or synthesis.")
            
            return None

        except Exception as e:
            logger.warning(f"[Local Dataset Error] {e}")
            
            return None



class WebResearch:
    @staticmethod
    async def fetch_json(url, params=None):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    return await response.json() if response.status == 200 else None
        except Exception as e:
            logger.warning(f"[Web JSON Error] {url}: {e}")
            return None

    @staticmethod
    async def fetch_html(url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    return await response.text() if response.status == 200 else None
        except Exception as e:
            logger.warning(f"[Web HTML Error] {url}: {e}")
            return None

    @staticmethod
    async def fetch_sources(query):
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in research_cache:
            return {"summary": research_cache[cache_key]}

        clean_query = WebResearch.preprocess_query(query)
        logger.info(f"[Preprocessed Query] {clean_query}")

        raw = {}

        if config.DUCKDUCKGO_RESEARCH_ENABLED:
            html_text = await WebResearch.fetch_html(DUCKDUCKGO_HTML + clean_query.replace(" ", "+"))
            if html_text:
                soup = BeautifulSoup(html_text, 'html.parser')
                snippet = soup.select_one('a.result__a')
                if snippet:
                    raw["duckduckgo"] = snippet.text.strip()

        if config.WIKIPEDIA_RESEARCH_ENABLED:
            data = await WebResearch.fetch_json(WIKIPEDIA_API + clean_query.replace(" ", "%20"))
            if data and "extract" in data:
                raw["wikipedia"] = data["extract"]

        if config.FREE_DICTIONARY_RESEARCH_ENABLED:
            html_text = await WebResearch.fetch_html(FREE_DICTIONARY_API + clean_query)
            if html_text:
                soup = BeautifulSoup(html_text, 'html.parser')
                block = soup.select_one('.ds-single')
                if block:
                    raw["dictionary"] = block.text.strip()

        if config.OPENLIBRARY_RESEARCH_ENABLED:
            data = await WebResearch.fetch_json(OPENLIBRARY_API + clean_query)
            if data and data.get("docs"):
                raw["openlibrary"] = data["docs"][0].get("title")

        combined_raw_text = "\n".join([v for v in raw.values() if isinstance(v, str)])
        synthesized_response = WebSemanticSynthesizer.synthesize_response(combined_raw_text, query)

        if synthesized_response and "still researching" not in synthesized_response.lower():
            research_cache[cache_key] = synthesized_response
            return {"summary": synthesized_response}

        for val in raw.values():
            if val:
                return {"summary": val}

        return None

    @staticmethod
    def preprocess_query(query):
        prefixes = ["what is", "who is", "tell me about", "give me information on", "explain", "define"]
        query = query.lower().strip()
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
                break
        return query

class APIResearch:
    @staticmethod
    def query(query, intent):
        try:
            # 🔒 Ultimate Global Kill Switch for All API Researching
            if not config.API_RESEARCH_ENABLED:
                logger.warning("[BLOCKED] API Research globally disabled via config.API_RESEARCH_ENABLED = False")
                return None

            def fallback_provider(current):
                priority = ["openai", "claude", "mistral", "gemini", "huggingface"]
                return next((p for p in priority if p != current and getattr(config, f"{p.upper()}_API", False)), None)

            provider = None
            if config.OPEN_AI_API:
                provider = "openai"
            elif config.CLAUDE_API:
                provider = "claude"
            elif config.MISTRAL_API:
                provider = "mistral"
            elif config.GEMINI_API:
                provider = "gemini"
            elif config.HUGGINGFACE_API:
                provider = "huggingface"

            if not provider:
                logger.warning("[AI PROVIDER ERROR] No AI provider flag set to True in SarahMemoryGlobals.py")
                return None

            result = send_to_api(query, provider=provider, intent=intent, tone="neutral", complexity="adult")
            if not result or not result.get("data"):
                fallback = fallback_provider(provider)
                if fallback:
                    logger.warning(f"[API] Primary {provider} failed, attempting fallback: {fallback}")
                    result = send_to_api(query, provider=fallback, intent=intent, tone="neutral", complexity="adult")

            if result and isinstance(result, dict):
                return {
                    "source": result.get("source", provider),
                    "intent": result.get("intent", intent),
                    "data": result.get("data", "")
                }
            return None

        except Exception as e:
            logger.warning(f"[API Fallback Error] {e}")
            return None


async def parallel_research(query):
    intent = classify_intent(query)

    async def local_task():
        if config.LOCAL_DATA_ENABLED:
            return LocalResearch.search(query, intent)
        return None

    async def web_task():
        if config.WEB_RESEARCH_ENABLED:
            try:
                return await WebResearch.fetch_sources(query)
            except Exception as e:
                logger.warning(f"[Parallel Web Error] {e}")
        return None

    async def api_task():
        # 🛡️ Hard block to prevent any API usage if master switch is off
        if not config.API_RESEARCH_ENABLED:
            logger.info("[API TASK BLOCKED] API_RESEARCH_ENABLED is set to False. Skipping API call.")
            return None

        # Wrap sync call to query() in async-safe thread
        return await asyncio.to_thread(APIResearch.query, query, intent)

    results = await asyncio.gather(local_task(), web_task(), api_task())

    results = [r for r in results if r]

    for r in results:
        if isinstance(r, dict) and r.get("data"):
            flat_data = r["data"]
            if isinstance(flat_data, dict) and "summary" in flat_data:
                flat_data = flat_data["summary"]
            return {"source": r.get("source", "unknown"), "intent": intent, "data": flat_data}
        elif isinstance(r, dict) and r.get("results"):
            return {"source": "parallel-local", "intent": intent, "data": r["results"][0]["snippet"]}
        logger.info(f"[DEBUG] get_research_data returning result: {results}")
    return {
        "source": "none",
        "intent": intent,
        "data": "Sorry, I was unable to find any reliable info using all sources."
    }


def get_research_data(query):
    logger.info(f"[DEBUG] Entered get_research_data with query: {query}")
    return asyncio.run(parallel_research(query))
