#!/usr/bin/env python3
"""Create simple synthetic scene-style PNG images for junior English testing."""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Create output directory
out_dir = Path("test_images_scene")
if out_dir.exists():
    import shutil
    shutil.rmtree(out_dir)
out_dir.mkdir()

# Scene definitions: (filename, width, height, draw_function_name)
scenes = [
    (
        "01_classroom.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h)], fill=(220, 240, 255)),  # Light blue sky
            draw.rectangle([(0, h*0.6), (w, h)], fill=(144, 238, 144)),  # Green floor
            draw.rectangle([(30, 120), (370, 260)], outline="black", width=3),  # Classroom outline
            draw.line([(200, 120), (200, 260)], fill="black", width=2),  # Center line
            [draw.rectangle([(x, 130), (x+30, 160)], fill="yellow", outline="black", width=1) for x in [50, 100, 150, 250, 300, 350]],  # Windows
            draw.rectangle([(40, 200), (360, 250)], outline="black", width=2),  # Board
            draw.text((150, 270), "A classroom", fill="black"),
        ),
    ),
    (
        "02_park.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h*0.4)], fill=(135, 206, 235)),  # Blue sky
            draw.rectangle([(0, h*0.4), (w, h)], fill=(34, 139, 34)),  # Green grass
            [draw.polygon([(100, h*0.4), (80, h*0.2), (120, h*0.2)], fill="darkgreen") for _ in [1]],  # Tree
            draw.circle((150, 150), 30, fill="red"),  # Ball
            draw.text((150, 270), "A park with trees", fill="black"),
        ),
    ),
    (
        "03_playground.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h*0.5)], fill=(176, 224, 230)),  # Light blue sky
            draw.rectangle([(0, h*0.5), (w, h)], fill=(210, 180, 140)),  # Sandy ground
            draw.rectangle([(50, 120), (80, 220)], fill="brown"),  # Swing pole
            draw.circle((65, 140), 15, fill="yellow", outline="black"),  # Swing seat
            draw.polygon([(200, 150), (170, 220), (230, 220)], fill="orange"),  # Slide
            draw.text((120, 270), "A playground", fill="black"),
        ),
    ),
    (
        "04_home.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h*0.4)], fill=(135, 206, 235)),  # Blue sky
            draw.rectangle([(0, h*0.4), (w, h)], fill=(144, 238, 144)),  # Green grass
            draw.rectangle([(80, 120), (320, 260)], fill=(210, 180, 140), outline="brown", width=3),  # House
            draw.polygon([(80, 120), (200, 40), (320, 120)], fill="darkred"),  # Roof
            [draw.rectangle([(x, 150), (x+40, 190)], fill="cyan", outline="brown", width=2) for x in [100, 240]],  # Windows
            draw.rectangle([(170, 220), (230, 260)], fill="brown", outline="black", width=2),  # Door
            draw.text((130, 275), "A house", fill="black"),
        ),
    ),
    (
        "05_school_bus.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h*0.5)], fill=(135, 206, 235)),  # Blue sky
            draw.rectangle([(0, h*0.5), (w, h)], fill=(200, 200, 200)),  # Gray road
            draw.rectangle([(50, 100), (350, 200)], fill="yellow", outline="black", width=3),  # Bus body
            draw.rectangle([(60, 110), (150, 150)], fill="cyan", outline="black", width=1),  # Window 1
            draw.rectangle([(160, 110), (250, 150)], fill="cyan", outline="black", width=1),  # Window 2
            draw.rectangle([(260, 110), (340, 150)], fill="cyan", outline="black", width=1),  # Window 3
            [draw.circle((x, 220), 20, fill="black") for x in [100, 300]],  # Wheels
            draw.text((130, 260), "A school bus", fill="black"),
        ),
    ),
    (
        "06_apple_tree.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h*0.4)], fill=(135, 206, 235)),  # Blue sky
            draw.rectangle([(0, h*0.4), (w, h)], fill=(144, 238, 144)),  # Green grass
            draw.rectangle([(170, 160), (230, 260)], fill="saddlebrown"),  # Trunk
            draw.ellipse([(80, 80), (320, 200)], fill="darkgreen", outline="darkgreen", width=2),  # Canopy
            [draw.circle((x, y), 12, fill="red") for x, y in [(140, 110), (200, 90), (260, 120), (180, 140), (240, 140)]],  # Apples
            draw.text((130, 270), "An apple tree", fill="black"),
        ),
    ),
    (
        "07_child_reading.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h)], fill=(240, 240, 240)),  # Light background
            draw.ellipse([(150, 50), (250, 120)], fill="wheat"),  # Head
            draw.rectangle([(160, 120), (240, 200)], fill="lightblue"),  # Shirt
            draw.rectangle([(150, 140), (200, 180)], fill="wheat"),  # Arm
            draw.rectangle([(120, 180), (280, 220)], fill="lightyellow", outline="black", width=2),  # Book
            [draw.line([(x, 190), (x+10, 190)], fill="black", width=1) for x in range(130, 270, 20)],  # Book lines
            draw.text((120, 260), "A child reading", fill="black"),
        ),
    ),
    (
        "08_cat.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h)], fill=(240, 255, 240)),  # Mint background
            draw.ellipse([(100, 100), (300, 200)], fill="orange", outline="black", width=2),  # Body
            draw.circle((250, 80), 40, fill="orange", outline="black", width=2),  # Head
            [draw.polygon([(240, 40), (250, 20), (260, 40)], fill="orange", outline="black") for _ in [1]],  # Left ear
            [draw.polygon([(280, 40), (290, 20), (300, 40)], fill="orange", outline="black") for _ in [1]],  # Right ear
            draw.circle((240, 80), 5, fill="black"),  # Left eye
            draw.circle((270, 80), 5, fill="black"),  # Right eye
            draw.circle((255, 95), 4, fill="black"),  # Nose
            draw.text((140, 250), "A cat", fill="black"),
        ),
    ),
    (
        "09_dog.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h)], fill=(255, 250, 240)),  # Floral white background
            draw.ellipse([(80, 120), (320, 200)], fill="brown", outline="black", width=2),  # Body
            draw.circle((300, 80), 45, fill="brown", outline="black", width=2),  # Head
            draw.polygon([(310, 40), (330, 20), (345, 35)], fill="brown", outline="black"),  # Ear 1
            draw.polygon([(335, 35), (360, 15), (360, 40)], fill="brown", outline="black"),  # Ear 2
            draw.circle((310, 85), 6, fill="black"),  # Eye
            draw.circle((325, 100), 5, fill="black"),  # Nose
            [draw.rectangle([(x, 200), (x+20, 260)], fill="brown", outline="black", width=1) for x in [100, 150, 220, 270]],  # Legs
            draw.text((140, 270), "A dog", fill="black"),
        ),
    ),
    (
        "10_sun_and_clouds.png",
        400,
        300,
        lambda draw, w, h: (
            draw.rectangle([(0, 0), (w, h)], fill=(135, 206, 235)),  # Sky blue
            draw.circle((350, 50), 40, fill="yellow", outline="orange", width=3),  # Sun
            draw.ellipse([(30, 80), (150, 140)], fill="white", outline="lightgray", width=2),  # Cloud 1
            draw.ellipse([(250, 150), (370, 210)], fill="white", outline="lightgray", width=2),  # Cloud 2
            draw.rectangle([(0, 250), (w, h)], fill=(144, 238, 144)),  # Green grass
            draw.text((130, 270), "Sunny day", fill="darkgreen"),
        ),
    ),
]

# Create images
try:
    for filename, w, h, draw_func in scenes:
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        try:
            draw_func(draw, w, h)
        except Exception as e:
            # If drawing fails, just fill with color and text
            draw.rectangle([(0, 0), (w, h)], fill=(200, 220, 240))
            draw.text((50, 150), filename.replace(".png", ""), fill="black")
        
        path = out_dir / filename
        img.save(path)
        print(f"Created {path.name} ({path.stat().st_size} bytes)")
except ImportError:
    print("ERROR: PIL (Pillow) not installed. Install with: pip install pillow")
    exit(1)

print(f"\n✓ Created {len(scenes)} simple scene-style test images in '{out_dir}'")
