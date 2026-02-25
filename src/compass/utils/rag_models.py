from pydantic import BaseModel, Field
from typing import List


class SubQuery(BaseModel):
    intent: str = Field(
        description="La intención específica de esta sub-búsqueda (Ej: ¿PUEDO HACERLO?, DINERO_LIMITES, etc.)"
    )
    keywords: List[str] = Field(
        description="Lista de palabras clave corporativas extraidas y expandidas para esta sub-búsqueda específica. NO incluir verbos auxiliares, cortesías, ni ruido."
    )


class SearchQuery(BaseModel):
    is_standalone: bool = Field(
        description="¿La pregunta depende de contexto anterior continuo en la conversación?"
    )
    original_intent: str = Field(
        description="La intención general de la consulta original."
    )
    sub_queries: List[SubQuery] = Field(
        description="Lista de sub-búsquedas independientes. Si la pregunta del usuario involucra múltiples temas (ej. un tope de gastos y también qué hacer si se pierde la factura), divídela en varias sub-búsquedas. Si es un solo tema, el arreglo tendrá 1 elemento."
    )