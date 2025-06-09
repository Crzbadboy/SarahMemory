# SarahCanvas.py
# SarahMemory Creative Art Engine - Generate AI-assisted artwork based on user prompts or instructions

import os
import datetime
import random
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

CANVAS_DIR = os.path.join("data", "canvas")
if not os.path.exists(CANVAS_DIR):
    os.makedirs(CANVAS_DIR)

COLORS = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 255, 255)]


def create_blank_canvas(width=512, height=512, color=(0, 0, 0)):
    return PIL.Image.new("RGB", (width, height), color=color)


def draw_prompt_on_canvas(image, prompt):
    draw = PIL.ImageDraw.Draw(image)
    font_path = os.path.join("data", "fonts", "arial.ttf")
    if not os.path.exists(font_path):
        font = PIL.ImageFont.load_default()
    else:
        font = PIL.ImageFont.truetype(font_path, 18)
    draw.text((10, 10), prompt, font=font, fill=random.choice(COLORS))
    return image


def save_canvas_image(image, tag="output"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{tag}_{timestamp}.png"
    path = os.path.join(CANVAS_DIR, filename)
    image.save(path)
    return path


def generate_ai_art(prompt="My first AI canvas"):
    image = create_blank_canvas()
    image = draw_prompt_on_canvas(image, prompt)
    saved_path = save_canvas_image(image, tag="ai_art")
    print(f"[SarahCanvas] Art saved to {saved_path}")


if __name__ == '__main__':
    generate_ai_art("Hello from SarahCanvas!")
