def split_text(text: str, chunk_size: int = 300, overlap: int = 200) -> list[str]:
    text = " ".join(text.split())  # tidy up spaces and line breaks
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # try to cut at a space, not in the middle of a word
            space = text.rfind(" ", start, end)
            if space > start + chunk_size // 2:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks