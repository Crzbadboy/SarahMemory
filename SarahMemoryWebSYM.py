#!/usr/bin/env python3
# ============================================
# SarahMemoryWebSYM.py v7.0 by Brian Lee Baros
# Description: Semantic synthesis of web-based content
# PATCH: Enterprise-grade Mathematical and Financial Brain Upgrade v6.6.33
# PATCH POINT: Corrected normalize_math_query logic, Added full fuzzy recall priority before API
# ============================================

import re
import html
import logging
import SarahMemoryGlobals as config
from SarahMemoryDatabase import search_answers  # ✅ New: Use fuzzy search first before calculating

logger = logging.getLogger("WebSYM")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# ✅ Expanded static fallback definitions
WEBSTER_STATIC = {
    "pi": "Pi is approximately 3.14159.",
    "microsoft": "Microsoft is a major software company founded by Bill Gates.",
    "elon musk": "Elon Musk is the CEO of Tesla and SpaceX.",
    "spacex": "SpaceX is an aerospace company founded by Elon Musk.",
    "bill gates": "Bill Gates is the co-founder of Microsoft and a philanthropist.",
    "python": "Python is a high-level programming language known for its readability.",
    "bitcoin": "Bitcoin is a decentralized digital cryptocurrency.",
    "starlink": "Starlink is a satellite internet constellation operated by SpaceX."
}

MATH_SYMBOLS = {
    "+": "plus", "-": "minus", "*": "times", "/": "divided by", "%": "percent", "^": "power", "√": "square root"
}

WORD_NUMBER_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1000000
}

NUMBER_WORD_MAP = {str(v): k for k, v in WORD_NUMBER_MAP.items() if v <= 1000}

UNIT_CONVERSION = {
    "inch": 1/12, "inches": 1/12, "centimeter": 0.0328084, "centimeters": 0.0328084,
    "meter": 3.28084, "meters": 3.28084, "foot": 1, "feet": 1
}

class WebSemanticSynthesizer:
    @staticmethod
    def strip_code(text):
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'(https?|ftp|www)\S+', '', text)
        text = re.sub(r'\{.*?\}', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,:;!?\'\-]', '', text)
        return text.strip()

    @staticmethod
    def compress_sentences(text):
        lines = text.split("\n")
        return [line.strip() for line in lines if len(line.strip()) > 20]

    @staticmethod
    def synthesize_response(content, query=""):
        import logging
        research_path_logger = logging.getLogger("ResearchPathLogger")
        logger.info(f"[DEBUG] Synthesizing Web Content: {content[:200]}")
        query = query.lower().strip()

        # Check if math
        if WebSemanticSynthesizer.is_math_query(query):
            return WebSemanticSynthesizer.sarah_calculator(query, query)

        # ✅ Try Fuzzy DB Lookup
        from SarahMemoryDatabase import search_answers  # Import here to avoid circular import
        # ✅ Conditionally Try Fuzzy DB Lookup Only If Enabled
        if config.LOCAL_DATA_ENABLED:
            fuzzy = search_answers(query)
            if fuzzy:
                logger.info("[RECALL] Fuzzy match triggered.")
                research_path_logger.info(f"returning fuzzy logic")
                return fuzzy[0]

        # ✅ Use content from web/API
        if content and len(content.strip()) > 20:
            compressed = WebSemanticSynthesizer.compress_sentences(content)
            logger.info(f"[DEBUG] Compressed result: {compressed}")
            if compressed:
                return compressed[0]

        # 🔻 If everything fails, check static fallback last
        if query in WEBSTER_STATIC:
            return WEBSTER_STATIC[query]
        research_path_logger.info(f"Webster_Static failure")
        return "I couldn't find reliable information to return right now."

    @staticmethod
    def is_math_query(query):
        math_keywords = ["plus", "minus", "times", "divided", "percent", "point",
                         "square root", "sqrt", "power", "of", "+", "-", "*", "/", "^", "%", "area", "square", "$", "dollar"]
        return any(word in query.lower() for word in math_keywords)

    @staticmethod
    def sarah_calculator(query, original_query=""):
        # ✅ PATCH: Attempt Fuzzy Recall First
        fuzzy_hits = search_answers(query)
        if fuzzy_hits:
            logger.info(f"[RECALL] Fuzzy DB recall success.")
            return fuzzy_hits[0]

        # Else, try to locally solve it
        try:
            parsed = WebSemanticSynthesizer.normalize_math_query(query)
            logger.debug(f"[DEBUG] Parsed Query: {parsed}")
            result = eval(parsed)
            return WebSemanticSynthesizer.format_final_answer(result, original_query)
        except Exception as e:
            logger.warning(f"[WARN] Local Solve Failed: {e}")
            if config.API_RESEARCH_ENABLED:
                external_answer = WebSemanticSynthesizer.route_to_external_api(query)
                if external_answer and WebSemanticSynthesizer.validate_api_math_response(external_answer):
                    return WebSemanticSynthesizer.format_final_answer(external_answer, original_query)
            return "I'm sorry, I couldn't solve that problem right now."

    @staticmethod
    def route_to_external_api(query):
        logger.info("[API] Routing math query to external API...")
        from SarahMemoryAPI import send_to_openai  # Moved inside function to fix circular import
        result = send_to_openai(f"Solve this math precisely: {query}")
        if result and isinstance(result, dict):
            return result.get("data")
        return None

    @staticmethod
    def validate_api_math_response(answer):
        if not answer:
            return False
        if re.search(r'[0-9]', str(answer)) and any(op in str(answer) for op in ["+", "-", "*", "/", "="]):
            return True
        return False

    @staticmethod
    def normalize_math_query(query):
        query = query.lower()
        query = WebSemanticSynthesizer.replace_units_in_query(query)
        query = WebSemanticSynthesizer.replace_math_words(query)

        if not re.search(r'\d', query):
            query = WebSemanticSynthesizer.words_to_numbers(query)

        query = re.sub(r'\bpoint\b', '.', query)
        query = re.sub(r'[^0-9.+\-*/%^()]', ' ', query)
        query = re.sub(r'\s+', ' ', query)
        return query.strip()

    @staticmethod
    def replace_math_words(query):
        replacements = {
            "percent of": "* 0.01 *", "percent": "* 0.01", "plus": "+", "minus": "-",
            "times": "*", "divided by": "/", "divided": "/", "power of": "**",
            "square root of": "**0.5", "squared": "**2", "cubed": "**3", "sqrt": "**0.5"
        }
        for word, symbol in replacements.items():
            query = query.replace(word, symbol)
        return query

    @staticmethod
    def words_to_numbers(text):
        tokens = text.split()
        current = 0
        result = []
        for token in tokens:
            if token in WORD_NUMBER_MAP:
                scale = WORD_NUMBER_MAP[token]
                if scale in [100, 1000, 1000000]:
                    if current == 0:
                        current = 1
                    current *= scale
                else:
                    current += scale
            elif token in ['+', '-', '*', '/', '**', '%']:
                result.append(str(current))
                result.append(token)
                current = 0
            else:
                result.append(token)
        result.append(str(current))
        return ' '.join(result)

    @staticmethod
    def replace_units_in_query(query):
        for unit, factor in UNIT_CONVERSION.items():
            query = re.sub(rf'(\d+(\.\d+)?)\s*{unit}', lambda m: str(float(m.group(1)) * factor), query)
        return query

    @staticmethod
    def format_final_answer(answer, original_query=""):
        try:
            num = float(str(answer).replace(",", "").split()[0])
            if "percent" in original_query or "%" in original_query:
                return f"The answer is {num:.2f}%"
            elif "$" in original_query or "dollar" in original_query or "usd" in original_query:
                return f"The answer is ${num:,.2f}"
            elif num.is_integer():
                return f"The answer is {int(num):,}"
            else:
                return f"The answer is {num:,}"
        except Exception:
            return f"The answer is {answer}."
