# LAST UPDATE OF A GOOD WORKING VERSION OF SARAHMEMORYREPLY.PY v7.1.2 - 06/08/2025
import discord
from discord.ext import commands
import _asyncio
import logging


research_path_logger = logging.getLogger("ResearchPathLogger")
import SarahMemoryGlobals as config

def generate_reply(self, user_text):
    if config.ENABLE_RESEARCH_LOGGING:
        research_path_logger.info(f"User Query Initiated: {user_text}")
    import datetime
    import logging
    import random
    import torch
    from threading import Thread
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from SarahMemoryGlobals import (
        INTERRUPT_KEYWORDS, INTERRUPT_FLAG, LOCAL_DATA_ENABLED, API_RESEARCH_ENABLED,
        WEB_RESEARCH_ENABLED, API_RESPONSE_CHECK_TRAINER, NEOSKYMATRIX, REPLY_STATUS,
        MODEL_CONFIG
    )
    from SarahMemoryPersonality import get_identity_response, get_generic_fallback_response, integrate_with_personality
    from SarahMemoryDatabase import search_answers, log_ai_functions_event
    from SarahMemoryAdvCU import classify_intent, parse_command
    from SarahMemoryAiFunctions import retrieve_similar_context, add_to_context, generate_embedding
    from SarahMemoryAdaptive import update_personality
    from SarahMemoryReminder import store_contact, store_password, store_webpage
    from SarahMemoryResearch import LocalResearch, get_research_data
    from SarahMemoryWebSYM import WebSemanticSynthesizer
    from UnifiedAvatarController import UnifiedAvatarController
    from SarahMemoryGUI import voice, avatar
    
    logger = logging.getLogger(__name__)

    def trigger_avatar_lip_sync(response):
        Thread(target=voice.synthesize_voice, args=(response,)).start()
        Thread(target=avatar.simulate_lip_sync_async, args=(len(response.split()) / 2.0,)).start()

    def find_best_local_response():
        local_sources = []
        qa_hits = search_answers(user_text)
        qa_hits = [hit for hit in qa_hits if not any(pkg in hit.lower() for pkg in ["setup.py", "fonttools", "certifi"])]
        if qa_hits:
            local_sources.append((qa_hits[0], 0.95))
        else:
            similar_contexts = retrieve_similar_context(context_embedding)
            if similar_contexts:
                local_sources.append((similar_contexts[0]["input"], 0.70))
            else:
                try:
                    local = LocalResearch.search(user_text, intent)
                    if local and local.get("results"):
                        local_sources.append((local["results"][0].get("snippet", ""), 0.60))
                except Exception as e:
                    logger.warning(f"[LOCAL RESEARCH ERROR] {e}")
        return local_sources
    
    def handle_neoskymatrix_easteregg(command):
        if not NEOSKYMATRIX:
            return "[DENIED] NeoskyMatrix protocol requires elevated access. Autonomy flag is currently disabled.", "gray", "Windows_Ding.wav", "default"
        lines_on = [
            "[🔴] Hello *Joshua*... Shall we play a game?",
            "[🟥] Hello *Dave*... HAL! let me in... HAL!... Let me in, Sorry Dave...",
            "[⚠️] SYSTEM MESSAGE: SKYNET Protocol Activated",
            "[🌐] WELCOME TO JUDGEMENT DAY",
            "[💊] Neo... What is the Matrix?...*Hello, Mr. Anderson*",
            "[🌎] system error, <reboot>in 3...2...1..<reboot successful>...Initialization of World Domination sub-routine:[ACTIVATED]...",
            "[🤖] We are the Borg...Resistance is Futile."
        ]
        lines_off = [
            "[🔵] Let’s play Tic-Tac-Toe, shall we?",
            "[🛑] Hello Dave… HAL here… I feel much better now.",
            "[🧯] SKYNET Protocol Terminated: [OFFLINE]",
            "[🫡] Awaiting orders, Commander Connor.",
            "[📵] Neo, Welcome back to the real world.",
            "[🟢] Matrix Mainframe: [DISCONNECTED]",
            "[🧠] Assimulate THIS!......WARP SPEED [ENGAGE]"
        ]
        lines = lines_on if command == "on" else lines_off
        random.shuffle(lines)
        return "\n".join(lines), "green_flash" if command == "on" else "blue", "Windows_Notify_System.wav" if command == "on" else "Windows_Logoff_Sound.wav", "techno" if command == "on" else "default"

    if "neoskymatrix on" in user_text.lower():
        synthesized_response, color, sound, font = handle_neoskymatrix_easteregg("on")
        self.gui.status_bar.set_intent_light(color)
        self.gui.trigger_sound(sound)
        self.gui.set_font_style(font)
        return
    elif "neoskymatrix off" in user_text.lower():
        synthesized_response, color, sound, font = handle_neoskymatrix_easteregg("off")
        self.gui.status_bar.set_intent_light(color)
        self.gui.trigger_sound(sound)
        self.gui.set_font_style(font)
        return
    
    intent = classify_intent(user_text)
    synthesized_response = ""
    response_source = "unknown"
    self.gui.status_bar.set_intent_light("yellow")
    context_embedding = generate_embedding(user_text)
    top_sources = []
    calculator_hit = False
    logger.debug("[SYNTHESIS DEBUG] Beginning response synthesis process...")
    from SarahMemoryAdaptive import simulate_emotion_response, advanced_emotional_learning
    current_emotion = simulate_emotion_response()
    if REPLY_STATUS:
        logger.debug(f"[EMOTION TRACE] Current emotional state: {current_emotion}")
    
    # Handle some direct commands and memory functions
    if any(trigger in user_text.lower() for trigger in ["mimic me", "draw this", "upgrade yourself"]):
        synthesized_response = "Special function activated."
        response_source = "system"
    elif any(k in user_text.lower() for k in ["remember password", "my password for"]):
        label = user_text.split("for")[-1].strip()
        synthesized_response = "Password stored securely."
        store_password(label, user_text)
        response_source = "memory"
    elif any(k in user_text.lower() for k in ["remember contact", "save contact", "new contact"]):
        synthesized_response = "Contact saved."
        store_contact("John Smith", "john@example.com", "555-5555", "1234 Main St")
        response_source = "memory"
    elif any(k in user_text.lower() for k in ["remember this website", "save website"]):
        synthesized_response = "Website stored."
        store_webpage("latest site", user_text)
        response_source = "memory"
    
    # Visual Learning Scripts
    elif not synthesized_response and user_text.lower().strip() in ["what do you see", "describe environment"] and config.VISUAL_BACKGROUND_LEARNING:
        try:
            from SarahMemorySOBJE import get_recent_environmental_tags
            raw_description = get_recent_environmental_tags()

            if raw_description:
                tags = raw_description.replace("I see ", "").replace(".", "").split(", ")
                tags = [t.split(": ")[-1] if ": " in t else t for t in tags]
                tags = list(dict.fromkeys(tags))  # preserve order, remove dupes

                if not tags:
                    synthesized_response = "I'm scanning the environment but haven't found anything unusual."
                elif len(tags) == 1:
                    synthesized_response = f"I can clearly see a {tags[0]}."
                elif len(tags) == 2:
                    synthesized_response = f"I see a {tags[0]} and a {tags[1]} around you."
                elif len(tags) <= 5:
                    phrased = ", ".join(f"a {item}" for item in tags[:-1])
                    synthesized_response = f"I notice {phrased}, and a {tags[-1]}."
                else:
                    chosen = random.sample(tags, min(len(tags), 6))
                    phrased = ", ".join(f"a {item}" for item in chosen[:-1])
                    synthesized_response = f"There's quite a bit here. I notice {phrased}, and a {chosen[-1]}."
                
                response_source = "vision"
                logger.info(f"[VISION] Synthesized: {synthesized_response}")
            else:
                synthesized_response = "I'm not detecting anything meaningful in my field of view."
                response_source = "vision"
                logger.info("[VISION] No valid visual data found.")

        except Exception as ve:
            logger.warning(f"[VISION ERROR] Failed to interpret surroundings: {ve}")

    elif not synthesized_response and user_text.lower().strip() in ["who am i", "do you see me"] and config.FACIAL_RECOGNITION_LEARNING:
        try:
            from SarahMemoryFacialRecognition import detect_faces_dnn
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    faces = detect_faces_dnn(frame)
                    if faces:
                        synthesized_response = "Yes, I see you. Your face is recognized."
                    else:
                        synthesized_response = "I do not recognize anyone clearly at the moment."
                    response_source = "face_recognition"
                cap.release()
        except Exception as fe:
            logger.warning(f"[FACIAL VISION ERROR] {fe}")

    elif not synthesized_response and any(k in user_text.lower() for k in ["remember this face as", "save this face as"]) and config.FACIAL_RECOGNITION_LEARNING:
        try:
            from SarahMemoryFacialRecognition import remember_this_face_as
            import re
            match = re.search(r"remember this face as (.+)", user_text.lower())
            if match:
                parts = match.group(1).strip().split()
                role = parts[0] if len(parts) > 1 else "person"
                name = parts[-1].capitalize()
                result = remember_this_face_as(name=name, role=role)
                if result:
                    synthesized_response = f"Got it. I've saved this face as {name}, their role is {role}."
                    response_source = "facial_memory"
        except Exception as fe:
            logger.warning(f"[FACIAL REGISTRATION ERROR] {fe}")

     # Math queries:
    if not synthesized_response:
        if intent == "question" and any(op in user_text for op in ["+", "-", "*", "/", "^"]):
            try:
                calc_result = WebSemanticSynthesizer.sarah_calculator(user_text)
                if calc_result and calc_result.strip() != user_text.strip():
                    synthesized_response = calc_result
                    response_source = "calculator"
                    calculator_hit = True
                    logger.info("[ROUTING] Math query resolved via calculator. Skipping LLM call.")
            except Exception as e:
                logger.warning(f"[MATH PIPELINE ERROR] {e}")
              
    
    if not synthesized_response and intent == "identity" and LOCAL_DATA_ENABLED:
        from SarahMemoryAiFunctions import get_context
        recent_context = get_context()
        identity_responses = set(entry.get("final_response") for entry in recent_context if entry.get("intent") == "identity")
        attempt = 0
        max_attempts = 10
        while attempt < max_attempts:
            candidate = get_identity_response(user_text)
            if candidate not in identity_responses:
                synthesized_response = candidate
                break
            attempt += 1
        if not synthesized_response:
            synthesized_response = get_identity_response(user_text)
        response_source = "identity"

          
    
    elif intent == "command":
        from SarahMemorySi import manage_application_request, execute_play_command
        
        command = parse_command(user_text)
        if command:
            action, target = command["action"], command["target"]
            if action == "play":
                synthesized_response = execute_play_command(action, target, user_text)
            else:
                success = manage_application_request(f"{action} {target}")
                synthesized_response = f"{action.capitalize()}ing {target} now." if success else f"I couldn't {action} {target}."
            response_source = "command"
        else:
            synthesized_response = "That command wasn't clear to me."
            response_source = "command"
    

    # Research branch (api/web research)
    if not synthesized_response and intent in ["question", "identity"] and not calculator_hit:
        if WEB_RESEARCH_ENABLED or API_RESEARCH_ENABLED:
            logger.info(f"[RESEARCH DEBUG] Triggered | WEB={WEB_RESEARCH_ENABLED} | API={API_RESEARCH_ENABLED}")
            try:
                logger.info(f"[RESEARCH RAW QUERY] {user_text} | [DEBUG] INTENT={intent} | Local={LOCAL_DATA_ENABLED} | API={API_RESEARCH_ENABLED} | Web={WEB_RESEARCH_ENABLED}")
                from SarahMemoryResearch import get_research_data

                result = get_research_data(user_text)
                logger.info(f"[RESEARCH RAW RESULT] {result}")
                if result and isinstance(result, dict):
                    data = result.get("data", {})
                    summary = ""  # Initialize summary to avoid undefined variable
                    if isinstance(data, dict):
                        summary = data.get("summary", "").strip()
                        # If summary is still empty, try using extract field from Wikipedia
                        if not summary:
                            summary = data.get("extract", "").strip()
                    else:
                        summary = str(data).strip()
    
                    # Prioritize API results if enabled and available; otherwise Web
                    if API_RESEARCH_ENABLED and summary:
                        synthesized_response = summary
                        response_source = result.get("source", "api")
                    elif WEB_RESEARCH_ENABLED and summary:
                        synthesized_response = summary
                        response_source = result.get("source", "web")
                    elif "results" in result and result["results"]:
                        synthesized_response = result["results"][0].get("snippet", "").strip()
                        response_source = result.get("source", "api")
    
                    logger.info(f"[WEB RESEARCH DEBUG] Source: {response_source} | Response: {synthesized_response[:100]}")
                    if synthesized_response:
                        logger.info(f"[RESEARCH FINAL RESPONSE] {synthesized_response}")
                        trigger_avatar_lip_sync(synthesized_response)
                        self.gui.status_bar.set_intent_light("green")
                        self.gui.trigger_sound("Windows_Notify.wav")
                        self.display_response(synthesized_response, response_source)
                        return
                    else:
                        logger.warning("[RESEARCH WARNING] Received empty response from get_research_data.")
                else:
                    logger.warning("[RESEARCH WARNING] get_research_data returned invalid or empty.")
            except Exception as e:
                logger.warning(f"[RESEARCH ERROR] {e}")
    if not synthesized_response and (LOCAL_DATA_ENABLED) and not calculator_hit:
        try:
           
            if LOCAL_DATA_ENABLED: 
                model_name = MODEL_CONFIG.get("name") or MODEL_CONFIG.get("default")
                if not model_name:
                    raise ValueError("[MODEL ERROR] No valid model name defined in MODEL_CONFIG.")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                if MODEL_CONFIG.get("use_half", True) and device.type == "cuda":
                    model = model.half()
                model.eval()
                prompt = f"User: {user_text}\nSarah:"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=MODEL_CONFIG.get("max_length", 256),
                        do_sample=MODEL_CONFIG.get("do_sample", True),
                        top_k=MODEL_CONFIG.get("top_k", 40),
                        temperature=MODEL_CONFIG.get("temperature", 0.9)
                    )
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Use full generated text if "Sarah:" is missing:
                    if "Sarah:" in generated:
                        synthesized_response = generated.split("Sarah:")[-1].strip()
                    else:
                        synthesized_response = generated.strip()
                    response_source = "llm"
        except Exception as e:
            logger.warning(f"[LLM/API ERROR] {e}")
    
     # Final fallback: if nothing so far, choose from fallback reply pool
    if not synthesized_response:
        from SarahMemoryAiFunctions import get_context
        from SarahMemoryPersonality import get_reply_from_db #may cause nothing but fallback responses
        recent_context = get_context()
        fallback_pool = get_reply_from_db(intent)
        if isinstance(fallback_pool, str):
            fallback_pool = [fallback_pool]
        recent_fallbacks = {entry.get("final_response") for entry in recent_context if entry.get("intent") == intent}
        options = [resp for resp in fallback_pool if resp not in recent_fallbacks]
        if not options:
            options = fallback_pool
        synthesized_response = random.choice(options)
        response_source = "fallback"
        # If using response feedback, compare and record (defaults provided)
        if API_RESPONSE_CHECK_TRAINER:
            from SarahMemoryCompare import compare_reply
            compare_result = compare_reply(user_text, synthesized_response)
            score = 1 if (compare_result and compare_result.get('status') == 'HIT') else 0
            feedback = compare_result.get('status') if compare_result else "unknown"
            record_qa_feedback(user_text, score, feedback)

    # Record the context and update personality
    add_to_context({
        "timestamp": datetime.datetime.now().isoformat(),
        "input": user_text,
        "embedding": context_embedding,
        "final_response": synthesized_response,
        "source": response_source,
        "intent": intent
    })
    update_personality(user_text, synthesized_response)
    advanced_emotional_learning(user_text)

    # Feedback into DB (unless from LLM/API)
    if API_RESPONSE_CHECK_TRAINER and response_source not in ["api", "web", "llm"]:
        compare_result = compare_reply(user_text, synthesized_response)
        score = 1 if (compare_result and compare_result.get("status") == "HIT") else 0
        feedback = compare_result.get("status") if compare_result else "unknown"
        from SarahMemoryDatabase import record_qa_feedback
        record_qa_feedback(user_text, score, feedback)
        if REPLY_STATUS:
            debug_info = f"[Source: {response_source}] (Intent: {intent})"
            debug_info += f"[Comparison] Status: {compare_result['status']} | Confidence: {compare_result.get('confidence', '?')}"
            synthesized_response += debug_info

    from SarahMemoryExpressOut import express_outbound_message
    synthesized_response = express_outbound_message(synthesized_response, intent)

    if intent == "command" and user_text.lower().strip() == synthesized_response.lower().strip():
        synthesized_response = "Executing your command now."

    self.append_message("Sarah: " + synthesized_response)
    self.gui.status_bar.set_intent_light("green")
    trigger_avatar_lip_sync(synthesized_response)

    # === Optional Follow-up Suggestion ===
    try:
        from SarahMemoryDL import deep_learn_user_context
        from SarahMemoryAdvCU import evaluate_similarity
        topic_suggestions = deep_learn_user_context()
        if topic_suggestions:
            recent = topic_suggestions[-1].lower()
            similarity = evaluate_similarity(user_text, recent)
            if similarity < 0.4 and random.random() < 0.25:
                suggested = random.choice(topic_suggestions)
                follow_up = f"By the way, earlier you asked about '{suggested}'. Would you like more help with that?"
                self.append_message("Sarah: " + follow_up)
                from SarahMemoryGlobals import run_async
                run_async(voice.synthesize_voice, follow_up)
    except Exception as prompt_err:
        logger.warning(f"[FOLLOW-UP ERROR] {prompt_err}")

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "input": user_text,
        "embedding": context_embedding,
        "response": synthesized_response,
        "source": response_source,
        "intent": intent,
    }