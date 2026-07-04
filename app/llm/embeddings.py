from app.config import get_settings
from app.llm.client import OpenRouterClient


async def embed_text(text: str) -> list[float]:
    settings = get_settings()
    client = OpenRouterClient()
    return await client.embed(settings.embedding_model, text[:4000], dimensions=settings.embedding_dim)
