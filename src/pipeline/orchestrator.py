from pathlib import Path
from datetime import datetime
from src.pipeline.storyboarder import make_stub_storyboard
from src.pipeline.image_gen import generate_placeholder_image
from src.pipeline.pdf_builder import build_pdf

OUT = Path('out')
BOOKS = OUT / 'books'
DEBUG = OUT / 'debug'

def run_once(transcript: str) -> Path:
    story = make_stub_storyboard(transcript)

    image_paths_by_page = {}
    for page in story.pages:
        paths = []
        for i, panel in enumerate(page.panels, start=1):
            img_path = DEBUG / f'page{page.page}_panel{i}.png'
            generate_placeholder_image(panel.image_prompt, img_path)
            paths.append(img_path)
        image_paths_by_page[page.page] = paths

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_pdf = BOOKS / f'story_{ts}.pdf'
    build_pdf(story, image_paths_by_page, out_pdf)
    return out_pdf
