import math
import os

from google import genai
from google.genai import types

MODEL = "gemini-embedding-001"
DIMENSIONS = 768
BATCH_SIZE = 50

_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else vector


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = get_client().models.embed_content(
            model=MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=DIMENSIONS,
            ),
        )
        vectors.extend(_normalize(e.values) for e in result.embeddings)
    return vectors