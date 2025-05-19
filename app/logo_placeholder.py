import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

def create_logo():
    """Create a basic placeholder logo for Fuzzy Diabetes"""
    # Create a white image
    width, height = 500, 300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    
    # Draw blue rectangle with rounded corners
    draw.rounded_rectangle([(50, 50), (450, 250)], fill="#0066FF", radius=20)
    
    # Try to use a good font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 60)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((100, 100), "Fuzzy", fill="white", font=font)
    draw.text((100, 170), "Diabetes", fill="white", font=font)
    
    # Save the image
    logo_path = os.path.join(os.path.dirname(__file__), "fuzzy_diabetes_logo.png")
    image.save(logo_path)
    print(f"Logo saved as {logo_path}")

if __name__ == "__main__":
    create_logo() 