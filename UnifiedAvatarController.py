#!/usr/bin/env python3
"""
UnifiedAvatarController.py <Version #7.1.2 Enhanced> <Author: Brian Lee Baros> rev. 060920252100
Description:
  This unified controller coordinates all avatar-related actions across the SarahMemory system.
  It processes voice or text commands to create or modify the avatar, integrates external API calls
  when local data is not available, and provides voice feedback via the TTS module.
Enhancements (v6.4):
  - Upgraded version header.
  - Improved auto-switching mechanism for design requests.
  - Detailed logging and caching of design information.
Notes:
  The controller splits functionality among modules (avatar display, voice synthesis, intent classification,
  and research) and is designed to run live within the GUI or in test mode.
"""

import logging
import time
import threading
import random
import os
import subprocess

# Import modules holding the distributed functionalities
import SarahMemoryAvatar as avatar_module
import SarahMemoryVoice as tts_module
import SarahMemoryAiFunctions as ai_functions
import SarahMemoryAdvCU as advcu_module
import SarahMemorySynapes as synapes_module
import SarahMemoryResearch as research_module
import SarahMemorySoftwareResearch as soft_research_module
import SarahMemoryGlobals as globals_module

# MOD: Define a simple asynchronous helper if run_async is not available.
try:
    run_async = globals_module.run_async
except AttributeError:
    def run_async(func, *args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()

# Setup logger for the unified controller
logger = logging.getLogger("UnifiedAvatarController")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# --------------------------
# Auto-Switching Mechanism
# --------------------------
class AutoSwitchingMechanism:
    def __init__(self):
        self.local_cache = {}

    def process_design_request(self, request_description):
        request_key = str(request_description)
        logger.info(f"Processing design request: {request_description}")
        if request_key in self.local_cache:
            logger.info("Local design information found in cache.")
            return self.local_cache[request_key]
        local_result = self.lookup_local_design(request_description)
        if local_result:
            self.local_cache[request_key] = local_result
            return local_result
        logger.info("Local data not found; querying external API...")
        api_result = synapes_module.compose_new_module(request_description)
        if api_result and "error" not in api_result.lower():
            self.local_cache[request_key] = api_result
            self.log_design_info(request_key, api_result)
            return api_result
        else:
            logger.error("API did not return valid design info.")
            return None

    def lookup_local_design(self, request_description):
        return None

    def log_design_info(self, request_description, design_info):
        logger.info(f"Logging design info: '{request_description}' -> {design_info}")

# --------------------------
# Unified Avatar Controller
# --------------------------
class UnifiedAvatarController:
    def __init__(self):
        self.auto_switch = AutoSwitchingMechanism()
        self.avatar = avatar_module
        self.tts = tts_module
        self.ai = ai_functions
        logger.info("UnifiedAvatarController initialized.")

    def create_avatar(self, design_request):
        import traceback

        if isinstance(design_request, str):
            design_request = {"request": design_request}
        logger.info(f"Creating avatar with request: '{design_request}'")

        design_info = self.auto_switch.process_design_request(design_request)

        if not isinstance(design_info, dict):
            logger.warning("Design info returned is not a valid dictionary. Using default values.")
            design_info = {
                "engine": "blender",
                "object_type": "dinosaur",
                "parameters": {
                    "location": (0, 0, 0),
                    "rotation": (0, 0, 0),
                    "scale": (1, 1, 1),
                    "material": "DefaultDinoMaterial"
                }
            }

        try:
            selected_engine = design_info.get("engine", "Blender")
            blend_file = os.path.join(globals_module.AVATAR_MODELS_DIR, "Sarah.blend")
            output_image = os.path.join(globals_module.AVATAR_DIR, "avatar_rendered.jpg")
            logger.debug(f"Calling Blender render: {blend_file} → {output_image}")
            success = launch_blender_avatar_render(blend_file, output_image)
            if success:
                self.tts.synthesize_voice("My avatar has been updated.")
                if hasattr(globals_module, 'avatar_panel_instance'):
                    try:
                        globals_module.avatar_panel_instance.update_avatar()
                        logger.info("AvatarPanel GUI refresh triggered.")
                    except Exception as e:
                        logger.warning(f"AvatarPanel refresh failed: {e}")
                logger.info("✅ Rendered 3D avatar updated.")
                try:
                    current_emotion = self.avatar.get_avatar_emotion()
                    self.avatar.set_avatar_expression(current_emotion)
                    self.avatar.update_avatar_expression(current_emotion)
                    self.avatar.simulate_lip_sync_async(2.0)
                except Exception as e:
                    logger.warning(f"Avatar sync after render failed: {e}")
            else:
                logger.warning("⚠️ Failed to render 3D avatar. Render function returned False.")
        except Exception as e:
            logger.error(f"❌ Exception during avatar rendering pipeline: {e}")
            traceback.print_exc()

    def modify_avatar(self, modification_command):
        logger.info(f"Modifying avatar with command: '{modification_command}'")
        command_lower = modification_command.lower()
        if "color" in command_lower:
            colors = ["red", "blue", "green", "yellow", "purple", "orange"]
            desired_color = next((word for word in command_lower.split() if word in colors), None)
            if desired_color:
                logger.info(f"Changing avatar color to {desired_color}.")
                self.tts.synthesize_voice(f"Changing my color to {desired_color}.")
                try:
                    if hasattr(self.avatar, "update_avatar_color"):
                        self.avatar.update_avatar_color(desired_color)
                    else:
                        logger.info("Simulated avatar color update.")
                except Exception as e:
                    logger.error(f"Error updating avatar color: {e}")
                    self.tts.synthesize_voice("I could not update my color.")
            else:
                logger.info("No valid color detected in command.")
                self.tts.synthesize_voice("I did not understand the color change request.")
        else:
            logger.info("Modification command not recognized.")
            self.tts.synthesize_voice("I did not understand that modification command.")

    def avatar_speak(self, message):
        logger.info(f"Avatar speaking: {message}")
        self.tts.synthesize_voice(message)
        try:
            current_emotion = self.avatar.get_avatar_emotion()
            logger.info(f"Syncing avatar emotion: {current_emotion}")
            self.avatar.set_avatar_expression(current_emotion)
            self.avatar.update_avatar_expression(current_emotion)
            self.avatar.simulate_lip_sync_async(len(message.split()) / 2.0)
        except Exception as e:
            logger.warning(f"Avatar emotion trigger failed: {e}")

def launch_blender_avatar_render(blend_file_path, output_image_path=None):
    try:
        blender_path = r"C:\Program Files\Blender Foundation\Blender 4.4\blender-launcher.exe"
        blender_script = f"""
import bpy
bpy.ops.wm.open_mainfile(filepath=r"{blend_file_path}")
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
scene = bpy.context.scene
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.eevee.taa_render_samples = 16
scene.eevee.use_soft_shadows = False
scene.eevee.use_bloom = False
scene.eevee.use_motion_blur = False
scene.frame_set(1)
scene.render.image_settings.file_format = 'JPEG'
scene.render.filepath = r"{output_image_path}"
bpy.ops.render.render(write_still=True)
"""
        temp_script = os.path.join(globals_module.SANDBOX_DIR, "temp_render_script.py")
        with open(temp_script, "w") as f:
            f.write(blender_script)

        result = subprocess.run(
            [blender_path, "--background", "--python", temp_script],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Blender render failed:\n{result.stderr}")
            return False
        else:
            logger.info("✅ Blender rendered avatar successfully.")
            return True

    except Exception as e:
        logger.error(f"❌ Exception during Blender render: {e}")
        return False

def main():
    controller = UnifiedAvatarController()
    test_commands = [
        "Create your avatar as a talking dinosaur",
        "Change your color to yellow",
        "Change your color to blue"
    ]
    for cmd in test_commands:
        logger.info(f"Processing test command: {cmd}")
        if "create your avatar" in cmd.lower():
            controller.create_avatar(cmd)
        elif "change your" in cmd.lower():
            controller.modify_avatar(cmd)
        time.sleep(3)

    run_async(controller.listen_for_commands)
    while True:
        time.sleep(5)

if __name__ == '__main__':
    main()
