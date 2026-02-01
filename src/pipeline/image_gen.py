from PIL import Image, ImageDraw
from pathlib import Path

def generate_placeholder_image(text: str, out_path: Path, size=(1024, 768)) -> Path:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)

    # bold border
    draw.rectangle([10, 10, size[0]-10, size[1]-10], outline="black", width=6)

    # center-ish text (simple)
    msg = text[:80] + ("..." if len(text) > 80 else "")
    draw.text((40, size[1]//2 - 10), msg, fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
