"""
Configuración y modelos de datos para el procesamiento de documentos de políticas.
"""

from dataclasses import dataclass, field
from typing import List, Dict

import tiktoken

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

SERVICE_ACCOUNT_FILE = (
    r"C:\Users\Lenovo\Desktop\Marker\SA-daf-aip-singularity-comp-sb.json"
)

# Configuración Drive
DRIVE_FOLDER_ID = ""
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

# Tokenizer global
tokenizer = tiktoken.get_encoding("cl100k_base")


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class ChunkingConfig:
    """Configuración para el chunking de documentos."""

    min_tokens: int = 100
    target_tokens: int = 500
    max_tokens: int = 600  # Hard limit
    overlap_tokens: int = 100
    filter_toc: bool = True
    filter_metadata_tables: bool = True

    # Contexto padre para embedding
    include_parent_context: bool = True
    parent_context_tokens: int = 150

    # Patrones de texto a limpiar de los chunks (ruido residual)
    filter_patterns: List[str] = field(
        default_factory=lambda: [
            # === Patrones existentes: disclaimers de confidencialidad ===
            r"La información de este documento es INTERNA[^.]*\.",
            r"La información contenida en este documento es INTERNA[^.]*\.",
            r"Ninguna de sus partes puede ser circulada[^.]*\.",
            r"sin autorización previa y por escrito\.",
            # === Headers/footers de página (texto plano) ===
            # Patrón completo: bloque Digital@FEMSA con metadata de página
            r"Digital@FEMSA\s*\n+HOJA\s*\n+\d+\s+de\s+\d+\s*\n+Nomenclatura\s*\n+ND-[^\n]+\n+[^\n]+\n+Versión:\s*\n+[\d.]+\s*\n+[^\n]+\n+Fecha de creación:\s*\n+[\d/]+\s*\n+Estatus\s*\n+Vigente\s*\n+Clasificación\s*\n+Interna",
            # Componentes individuales que escapan del patrón completo
            r"(?:^|\n)\s*HOJA\s*\n?\s*\d+\s+de\s+\d+\s*(?:\n|$)",
            r"(?:^|\n)\s*Nomenclatura\s*\n?\s*ND-[A-Z&]+-[A-Z]+-\d+\s*(?:\n|$)",
            r"(?:^|\n)\s*Versión:\s*\n?\s*[\d.]+\s*(?:\n|$)",
            r"(?:^|\n)\s*Fecha de creación:\s*\n?\s*[\d/]+\s*(?:\n|$)",
            r"(?:^|\n)\s*Estatus\s*\n?\s*Vigente\s*(?:\n|$)",
            r"(?:^|\n)\s*Clasificación\s*\n?\s*Interna\s*(?:\n|$)",
            # Headers sueltos de nombre de documento/empresa
            r"(?:^|\n)\s*Digital@FEMSA\s*(?:\n|$)",
            r"(?:^|\n)\s*Dirección Administración y Finanzas\s*(?:\n|$)",
        ]
    )

    # Palabras clave para detectar tablas de metadatos (usado en validación)
    metadata_keywords: List[str] = field(
        default_factory=lambda: [
            "elaboró",
            "revisó",
            "autorizó",
            "historial de versiones",
            "fecha de actualización",
            "resumen de cambios",
            "propietario del documento",
        ]
    )


@dataclass
class ChunkMetadata:
    """Metadatos para el resultado del chunking."""

    source_pdf: str
    strategy: str = "docling_section_based_v2"
    total_chunks: int = 0
    total_tokens: int = 0
    avg_tokens: float = 0.0
    min_chunk_tokens: int = 0
    max_chunk_tokens: int = 0
    pages_skipped: int = 0
    config: Dict = None


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================


def count_tokens(text: str) -> int:
    """Cuenta tokens usando tiktoken."""
    if not text or not text.strip():
        return 0
    return len(tokenizer.encode(text))


__all__ = [
    "SERVICE_ACCOUNT_FILE",
    "DRIVE_FOLDER_ID",
    "DRIVE_SCOPES",
    "tokenizer",
    "ChunkingConfig",
    "ChunkMetadata",
    "count_tokens",
]
