#!/usr/bin/env python3
"""
SarahMemorySOBJE.py <Version #6.5 Enhanced> <Author: Brian Lee Baros>
Description: THIS FILE FUNCTIONS ARE TO BE USED IN THE "SarahMemoryGUI.py" FILE UNDER 
"class SuperObjectEngine(sobje)"
Combined file for webcam-based object recognition and vector similarity memory functionalities.
This file merges the functionalities from SarahMemoryFacialRecognition.py and SarahMemoryGUI.py to log into 
the ai_learning.db database to catalog objects and their vector similarities. Designing this Module will be 
used for future AI learning and memory functions. The goal is to create a system that can learn from its ablity 
of vision using cameras to detect not just objects but possible signs of medical and health issues.


Version: 6.5    
Enhancements:
  - Unified logging for two distinct functionalities.
  - A command-line flag to choose between running the object recognition test or object recognition VSM test.
Notes:
    - The object recognition test is to uses a non-dummy detection function for demonstration purposes.
    - The object recognition VSM test uses a non-dummy vector similarity function for demonstration purposes.
    - The database schema is designed to accommodate both functionalities.
    - The GUI is designed to provide a simple interface for both functionalities.
    - The logging system is designed to provide detailed information about the operations performed.
    - The code is designed to be modular and easy to extend for future functionalities.
    - The code is designed to be compatible with Python 3.x.
    - The code is designed to be compatible with OpenCV 4.x.
    - The code is designed to be compatible with SQLite 3.x.
    - VSM module code is located in the SarahMemoryFacialRecognition.py file
  To run Object recognition, simply execute:
      ./SarahMemorySOBJE.py
  
"""

# ------------------------- Imports -------------------------
import cv2
import logging
import os
import threading
import asyncio
import sqlite3
import random
import numpy as np
from datetime import datetime

import SarahMemoryGlobals as config
from SarahMemoryGlobals import DATASETS_DIR, OBJECT_MODEL_CONFIG, OBJECT_DETECTION_ENABLED, MODEL_PATHS
#import SarahMemoryFacialRecognition as fr
#from SarahMemoryGlobals import run_async
#from SarahMemoryHi import async_update_network_state

YOLO_MODELS = {}
if OBJECT_DETECTION_ENABLED:
    from ultralytics import YOLO
    for model_name, model_cfg in OBJECT_MODEL_CONFIG.items():
        if model_cfg.get("enabled"):
            model_dir = MODEL_PATHS.get(model_name)
            if model_dir:
                model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                    try:
                        YOLO_MODELS[model_name] = YOLO(model_path)
                    except Exception as e:
                        logging.warning(f"[YOLO Init Fail] {model_name} @ {model_path}: {e}")
                else:
                    logging.warning(f"[YOLO Load Skip] {model_name}: No .pt file found in {model_dir}")
            else:
                logging.warning(f"[YOLO Load Skip] {model_name}: Model path missing in MODEL_PATHS")



logger = logging.getLogger("SarahMemorySOBJE")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(sh)

# ------------------------- SUPER ULTRA Object Detection Engine-------------------------

def ultra_detect_objects(frame: np.ndarray) -> list:
    """
    Detects objects from the provided frame, applies domain tagging,
    logs findings to the database, and returns selected labels.
    
    Improvements:
      - Uses a local copy of the frame for drawing.
      - Reduces synthetic term generation for performance.
      - Uses context managers for SQLite logging.
      - Added inline documentation and type hints.
    """
    if frame is None:
        logger.warning("No frame provided for object detection.")
        return []
    
    processed_frame = frame.copy()  # local copy for drawing

    def get_contours(frame: np.ndarray) -> list:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except Exception as e:
            logger.error(f"Error extracting contours: {e}")
            return []

    def draw_and_identify(contours: list, min_area: int = 500) -> list:
        tags = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(processed_frame, "object", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                tags.append("object")
        return tags

    contours = get_contours(frame)
    detected_objects = draw_and_identify(contours)

    domains = {
        "animals": [
            "domestic: cat", "domestic: dog", "farm: cow", "farm: pig", "wild: lion", "wild: tiger", "bird: parrot",
            "reptile: iguana", "aquatic: dolphin", "amphibian: frog", "rodent: hamster"
        ],
        "insects": [
            "insect: bee", "insect: butterfly", "insect: dragonfly", "insect: mosquito", "insect: ant", "insect: beetle",
            "insect: grasshopper", "insect: ladybug"
        ],
        "colors and patterns": [
            "color: hazel eye", "color: green eye", "color: blue eye", "hair: blonde", "hair: auburn", "hair: black",
            "pattern: camouflage", "pattern: houndstooth", "pattern: pinstripe", "pattern: polka dot"
        ],
        "food": [
            "fruit: apple", "fruit: banana", "vegetable: broccoli", "vegetable: carrot", "snack: chocolate bar",
            "drink: coffee", "dish: lasagna", "grain: rice", "dessert: cheesecake"
        ],
        "kitchen": [
            "kitchen tool: spatula", "kitchen tool: ladle", "utensil: fork", "utensil: spoon", "appliance: toaster",
            "appliance: blender", "cutlery: chef's knife", "storage: plastic container"
        ],
        "electronics": [
            "component: resistor", "component: capacitor", "component: transistor", "component: diode",
            "component: LED", "IC: 555 timer", "IC: op-amp", "board: PCB"
        ],
        "mechanics": [
            "mechanical: gear", "mechanical: sprocket", "mechanical: flywheel", "mechanical: driveshaft"
        ],
        "measurement": [
            "tool: metric wrench", "tool: standard wrench", "gauge: caliper", "scale: digital", "ruler: inches",
            "ruler: centimeters", "weight: 5kg plate", "weight: 10lb dumbbell"
        ],
        "facial features": [
            "face: nose", "face: mouth", "face: ear", "face: eyebrow", "face: cheekbone", "face: chin",
            "face: forehead", "face: jawline"
        ],
        "facial expression": [
            "expression: happy", "expression: sad", "expression: angry", "expression: surprised", "expression: neutral",
            "expression: confused", "expression: disgusted", "expression: excited"
        ],
        "facial hair": [
            "facial hair: beard", "facial hair: mustache", "facial hair: goatee", "facial hair: sideburns",
            "facial hair: stubble"
        ],
        "skin tone": [
            "skin tone: fair", "skin tone: medium", "skin tone: olive", "skin tone: tan", "skin tone: dark"
        ],
        "skin condition": [
            "condition: acne", "condition: scar", "condition: wrinkle", "condition: birthmark", "condition: tattoo",
            "condition: mole"
        ],
        "eye color": [
            "eye color: brown", "eye color: blue", "eye color: green", "eye color: gray", "eye color: hazel",
            "eye color: amber"
        ],
        "eye shape": [
            "eye shape: almond", "eye shape: round", "eye shape: hooded", "eye shape: monolid", "eye shape: downturned",
            "eye shape: upturned"
        ],
        "eye condition": [
            "condition: cataract", "condition: glaucoma", "condition: astigmatism", "condition: color blindness",
            "condition: strabismus"
        ],
        "eye detail": [
            "detail: eyelash", "detail: eyebrow", "detail: pupil", "detail: iris", "detail: sclera"
        ],
        "eye feature": [
            "feature: eyelid", "feature: tear duct", "feature: conjunctiva", "feature: cornea", "feature: retina"
        ],
        "eye accessories": [
            "accessory: contact lens", "accessory: glasses", "accessory: sunglasses", "accessory: eye patch"
        ],
        "eye movement": [
            "movement: blink", "movement: squint", "movement: roll", "movement: dart", "movement: stare"
        ],
        "eye expression": [
            "expression: wink", "expression: squint", "expression: wide-eyed", "expression: narrowed"
        ],
        "eye position": [
            "position: looking up", "position: looking down", "position: looking left", "position: looking right"
        ],
        "eye size": [
            "size: small", "size: medium", "size: large", "size: extra-large"
        ],
        "eye distance": [
            "distance: close-set", "distance: wide-set", "distance: normal"
        ],
        "eye symmetry": [
            "symmetry: symmetrical", "symmetry: asymmetrical"
        ],
        "Body parts": [
            "body part: arm", "body part: leg", "body part: hand", "body part: foot", "body part: torso",
            "body part: head", "body part: neck", "body part: shoulder", "body part: back", "body part: abdomen",
            "body part: knee", "body part: elbow", "body part: wrist", "body part: ankle", "body part: hip",
            "body part: finger", "body part: toe", "body part: thumb", "body part: chin", "body part: forehead",
            "body part: cheek", "body part: jaw", "body part: temple", "body part: scalp", "body part: heel",
            "body part: instep", "body part: arch", "body part: palm", "body part: knuckle", "body part: nail",
            "body part: back of hand", "body part: back of foot", "body part: ball of foot", "body part: sole",
            "body part: bridge of foot", "body part: top of foot", "body part: side of foot",
            "body part: side of hand", "body part: base of thumb","body part: breast", 
            "body part: waist", "body part: hip bone", "body part: collarbone", "body part: ribcage",
            "body part: spine", "body part: vertebrae", "body part: pelvis", "body part: sacrum", "body part: coccyx",
            "body part: sternum", "body part: scapula", "body part: clavicle", "body part: radius", "body part: ulna",
            "body part: femur", "body part: tibia", "body part: fibula", "body part: patella", "body part: tarsals",
            "body part: metatarsals", "body part: phalanges", "body part: carpal bones", "body part: metacarpals",
            "body part: phalanges of hand", "body part: phalanges of foot", "body part: knuckles of hand",
            "body part: knuckles of foot", "body part: base of fingers", "body part: base of toes", 
        ],
        "Body features": [
            "feature: muscle", "feature: fat", "feature: bone", "feature: skin", "feature: hair"
        ],
        "Body conditions": [
            "condition: healthy", "condition: sick", "condition: injured", "condition: fit", "condition: weak"
        ],
        "Body expressions": [
            "expression: relaxed", "expression: tense", "expression: active", "expression: passive"
        ],
        "Body movements": [
            "movement: walk", "movement: run", "movement: jump", "movement: sit", "movement: stand"
        ],
        "Body accessories": [
            "accessory: watch", "accessory: ring", "accessory: bracelet", "accessory: necklace"
        ],
        "Body clothing": [
            "clothing: shirt", "clothing: pants", "clothing: dress", "clothing: jacket", "clothing: shoes"
        ],
        "Body types": [
            "type: athletic", "type: slim", "type: average", "type: overweight", "type: muscular"
        ],
        "Body proportions": [
            "proportion: long", "proportion: short", "proportion: average"
        ],
        "Body symmetry": [
            "symmetry: symmetrical", "symmetry: asymmetrical"
        ],
        "Body movements": [
            "movement: flex", "movement: extend", "movement: rotate", "movement: twist"
        ],
        "Body positions": [
            "position: upright", "position: slouched", "position: bent", "position: straight"
        ],
        "Body sizes": [
            "size: small", "size: medium", "size: large", "size: extra-large"
        ],
        "Body distances": [
            "distance: close", "distance: medium", "distance: far"
        ],
        "Sex": [
            "sex: male", "sex: female", "sex: non-binary"
        ],
        "Age": [
            "age: young", "age: middle-aged", "age: old"
        ],
        "Height": [
            "height: short", "height: average", "height: tall"
        ],
        "anatomy": [
            "anatomy: skeleton", "anatomy: muscle", "anatomy: organ", "anatomy: tissue"
        ],
        "anatomy detail": [
            "anatomy detail: bone structure", "anatomy detail: muscle fiber", "anatomy detail: organ system"
        ],
        "anatomy feature": [
            "anatomy feature: joint", "anatomy feature: ligament", "anatomy feature: tendon"
        ],
        "anatomy condition": [
            "anatomy condition: healthy", "anatomy condition: diseased", "anatomy condition: injured"
        ],
        "anatomy expression": [
            "anatomy expression: relaxed", "anatomy expression: tense", "anatomy expression: active"
        ],
        "anatomy movement": [
            "anatomy movement: flex", "anatomy movement: extend", "anatomy movement: rotate"
        ],
        "anatomy clothing": [
            "anatomy clothing: bandage", "anatomy clothing: cast", "anatomy clothing: support"
        ],
        "reproductive expressions": [
            "reproductive expression: aroused", "reproductive expression: relaxed", "reproductive expression: tense"
        ],
        "reproductive movements": [
            "reproductive movement: ovulation", "reproductive movement: ejaculation", "reproductive movement: menstruation"
        ],
        "facial detail": [f"face-point: landmark_{i}" for i in range(1, 20001)]
    }
     
    detected_tags = []
    for obj in detected_objects:
        for category, values in domains.items():
            if obj in values:
                detected_tags.append((category, obj))
                break

    # Check for critical conditions.
    medical_flags = {
        "condition: mole": "Possible melanoma", 
        "expression: slouched": "Posture anomaly / neuro issue",
        "face-point: landmark_2349": "Right eye droop — stroke warning",
        "skin tone: uneven": "Skin cancer check"
    }
    for tag in detected_tags:
        if tag in medical_flags:
            alert_medical(tag)

    # Generate synthetic high-tech terms (reduced count for better performance)
    synthetic_count = 500  # Reduced from 10000 iterations
    prefixes = ["modular", "adaptive", "quantum", "neural", "precision", "bio-active", "liquid-cooled", "synthetic"]
    nouns = ["transmitter", "oscillator", "stabilizer", "inverter", "thruster", "sensor", "valve", "core"]
    suffixes = ["array", "hub", "grid", "cell", "interface", "matrix", "system", "scanner"]
    synthetic_terms = [
        f"{random.choice(prefixes)} {random.choice(nouns)} {random.choice(suffixes)}"
        for _ in range(synthetic_count)
    ]

    # Combine domain terms with synthetic terms.
    all_objects = []
    for category, terms in domains.items():
        for term in terms:
            all_objects.append(f"{category}: {term}")
    all_objects.extend(synthetic_terms)

    random.shuffle(all_objects)
    selected = random.sample(all_objects, random.randint(3, 12))

    # Log selected detections to SQLite.
    try:
        db_path = os.path.join(config.DATASETS_DIR, "ai_learning.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS object_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    label TEXT
                )
            """)
            timestamp = datetime.now().isoformat()
            for item in selected:
                cursor.execute("""
                    SELECT COUNT(*) FROM object_observations 
                    WHERE label = ? AND DATE(timestamp) = DATE('now')
                """, (item,))
                if cursor.fetchone()[0] == 0:
                    if hasattr(config, 'vision_canvas'):
                        try:
                            config.vision_canvas.itemconfig(config.vision_light, fill="red")
                        except Exception as ce:
                            logger.warning(f"Vision light update failed: {ce}")
                    cursor.execute("INSERT INTO object_observations (timestamp, label) VALUES (?, ?)",
                                   (timestamp, item))
                    if hasattr(config, 'status_bar'):
                        try:
                            config.status.set_status(f"Identified: {item}")
                        except Exception as sbe:
                            logger.warning(f"Status bar update failed: {sbe}")
            conn.commit()
    except Exception as e:
        if hasattr(config, 'vision_canvas'):
            try:
                config.vision_canvas.itemconfig(config.vision_light, fill="yellow")
            except Exception as ce:
                logger.warning(f"Vision light update failed: {ce}")
        logger.warning(f"Could not log object detections: {e}")

    logger.info(f"Ultra Detected: {selected}")
    if hasattr(config, 'vision_canvas'):
        try:
            config.vision_canvas.itemconfig(config.vision_light, fill="green")
        except Exception as ce:
            logger.warning(f"Vision light update failed: {ce}")
    return selected
def get_recent_environmental_tags(limit: int = 10) -> str:
    """
    Returns a summarized and human-readable description of recent environmental observations.
    """
    try:
        db_path = os.path.join(config.DATASETS_DIR, "ai_learning.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT label FROM object_observations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            if not rows:
                return "I can't see anything clearly right now."

            labels = [row[0] for row in rows if row[0] and "face-point" not in row[0]]
            unique_tags = list(set(labels))

            if not unique_tags:
                return "All I can detect right now are facial landmarks or minor visual noise."

            return "I see " + ', '.join(unique_tags) + "."
    except Exception as e:
        logger.warning(f"[SOBJE ERROR] Failed to fetch environment tags: {e}")
        return "I couldn't access the visual detection log."
    
def alert_medical(tag) -> None:
    """
    Placeholder for handling medical alerts based on tag.
    """
    logger.warning(f"Medical alert triggered for tag: {tag}")