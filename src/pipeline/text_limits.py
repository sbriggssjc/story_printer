def clamp_words(text: str, max_words: int) -> str:
    """
    Hard word cap so we never exceed the 2-page output budget.
    Keeps the first max_words words.
    """
    words = (text or "").split()
    if len(words) <= max_words:
        return text.strip()

    clipped = " ".join(words[:max_words]).strip()
    return clipped + " ..."
