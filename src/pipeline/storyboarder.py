from pydantic import BaseModel, Field
from typing import List

class Panel(BaseModel):
    caption: str
    image_prompt: str

class Page(BaseModel):
    page: int
    panels: List[Panel] = Field(default_factory=list)

class Storyboard(BaseModel):
    title: str
    pages: List[Page]

def make_stub_storyboard(transcript: str) -> Storyboard:
    title = "The Amazing Story"

    prompts = [
        "A cheerful kid in a sunny park begins an adventure, whimsical storybook fantasy, clean outlines, soft shading, comic panel, kid-friendly, no text",
        "A friendly animal appears with a funny surprise, whimsical storybook fantasy, clean outlines, soft shading, comic panel, kid-friendly, no text",
        "The kid makes a brave choice and smiles, whimsical storybook fantasy, clean outlines, soft shading, comic panel, kid-friendly, no text",
        "A silly obstacle blocks the path, whimsical storybook fantasy, clean outlines, soft shading, comic panel, kid-friendly, no text",
        "Friends work together to solve it, whimsical storybook fantasy, clean outlines, soft shading, comic panel, kid-friendly, no text",
        "Everyone celebrates in a bright happy scene, whimsical storybook fantasy, clean outlines, soft shading, comic panel, kid-friendly, no text",
    ]

    caps = [
        "Once upon a time, a kid had a big idea.",
        "Then something surprising popped up!",
        "They decided to be brave and try.",
        "A silly problem got in the way.",
        "But teamwork made it easy.",
        "And that’s how the adventure ended happily.",
    ]

    pages = [
        Page(page=1, panels=[Panel(caption=caps[i], image_prompt=prompts[i]) for i in range(0, 3)]),
        Page(page=2, panels=[Panel(caption=caps[i], image_prompt=prompts[i]) for i in range(3, 6)]),
    ]
    return Storyboard(title=title, pages=pages)
