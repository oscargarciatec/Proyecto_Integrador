"""
Utilidades de procesamiento de texto para documentos de políticas.
Incluye funciones de limpieza, detección de secciones, filtros y validación.
"""

import re
from typing import List, Tuple, Optional

from .settings import ChunkingConfig


# =============================================================================
# LIMPIEZA DE TEXTO
# =============================================================================


def clean_text(text: str, config: ChunkingConfig) -> str:
    """Limpia texto aplicando patrones de filtro configurados."""
    cleaned = text
    for pattern in config.filter_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return re.sub(r"\s+", " ", cleaned).strip()


# =============================================================================
# DETECCIÓN DE ELEMENTOS
# =============================================================================


def is_likely_list_item(text: str, config: ChunkingConfig = None) -> bool:
    """
    Detecta si un texto es un elemento de lista numerada, NO una sección.

    Usa patrones lingüísticos en lugar de listas de palabras:
    - Items de lista suelen empezar con minúscula después del número
    - Items de lista frecuentemente terminan con ":"
    - Secciones son ALL CAPS o Title Case corto
    """
    text = text.strip()

    # a) Formato letra-paréntesis: "a) texto", "b) texto"
    if re.match(r"^[a-z]\)\s", text, re.IGNORECASE):
        return True

    # b) Número + texto que termina en ":" → probablemente descripción de item
    if re.match(r"^\d+\.?\s+.+:\s*$", text):
        return True

    # c) Número + MINÚSCULA inmediatamente → item de lista, no sección
    # Ej: "1. por daño al equipo" (item) vs "1. INTRODUCCIÓN" (sección)
    if re.match(r"^\d+\.?\s+[a-záéíóúñ]", text):
        return True

    # d) Número + palabra corta en mayúscula + minúscula → item de lista
    # Ej: "1. El colaborador debe..." vs "1. OBJETIVO" (sección)
    if re.match(r"^\d+\.?\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[a-záéíóúñ]", text):
        return True

    # e) Contiene ":" en la primera mitad del texto → probablemente item
    if ":" in text[: len(text) // 2] and len(text) < 150:
        return True

    return False


def is_likely_section_title_line(text: str) -> bool:
    """Detecta si una línea es probablemente un título de sección."""
    text = (text or "").strip()
    if not text or len(text) > 150:
        return False
    # Detectar Anexo solo
    if re.match(r"^Anexo\s*[\dA-Z]*$", text, re.IGNORECASE):
        return True

    if text.isupper():
        return True
    if re.match(
        r"^[A-ZÁÉÍÓÚ][\wÁÉÍÓÚáéíóú]+(\s+[A-Za-zÁÉÍÓÚáéíóú]{1,4})*(\s+[A-ZÁÉÍÓÚ][\wÁÉÍÓÚáéíóú]+)+$",
        text,
    ):
        return True
    return False


# =============================================================================
# EXTRACCIÓN DE INFORMACIÓN DE SECCIÓN
# =============================================================================


def normalize_section_title(title: str) -> str:
    """Normaliza el título de una sección extrayendo palabras en mayúsculas."""
    title = (title or "").strip()
    if not title:
        return ""
    words = title.split()
    prefix = []
    for w in words:
        if w.isupper() and len(w) > 1:
            prefix.append(w)
        elif prefix:
            break
    return " ".join(prefix) if prefix else title


def extract_section_info(
    text: str, is_header_label: bool = False, config: ChunkingConfig = None
) -> Tuple[Optional[str], Optional[str], int]:
    """
    Extrae (section_number, section_title, hierarchy_level) de un texto.

    Si is_header_label=True (Docling lo marcó como SECTION_HEADER), confiamos
    en esa detección sin requerir validación de keywords.
    """
    text = text.strip()

    # Detectar Anexos (Anexo 1, Anexo 2, Anexo A, etc.)
    anexo_match = re.match(r"^(Anexo\s*[\dA-Z]*):?\s*(.*)$", text, re.IGNORECASE)
    if anexo_match:
        anexo_num = anexo_match.group(1).strip()
        anexo_title = (
            anexo_match.group(2).strip() if anexo_match.group(2) else anexo_num
        )
        return anexo_num, anexo_title, 1

    # Patrones de sección numerada: 1., 1.1, 1.1.1, 1.1.1.1
    patterns = [
        (r"^(\d+\.\d+\.\d+\.\d+)\.?\s+(.+)$", 4),
        (r"^(\d+\.\d+\.\d+)\.?\s+(.+)$", 3),
        (r"^(\d+\.\d+)\.?\s+(.+)$", 2),
        (r"^(\d+)\.?\s+(.+)$", 1),
    ]

    for pattern, level in patterns:
        match = re.match(pattern, text)
        if match:
            title = match.group(2).strip()

            # Filtros de falsos positivos para Nivel 1
            if level == 1:
                # Siempre filtrar items de lista (ej: "1. Por daño al equipo:")
                if is_likely_list_item(text, config):
                    return None, None, 0

                # Si Docling NO lo marcó como header, aplicar validación estricta
                if not is_header_label:
                    # Patrones que identifican títulos de sección válidos:
                    # - ALL CAPS: "INTRODUCCIÓN", "OBJETIVO", "VIGENCIA"
                    # - Titulo corto (1-4 palabras) sin minúsculas intermedias
                    # - No termina en ":" (eso sería descripción de item)

                    is_valid_section = False

                    # a) ALL CAPS → definitivamente sección
                    if title.isupper() and len(title) > 2:
                        is_valid_section = True

                    # b) Formato "Anexo X" o "ANEXO X"
                    elif re.match(r"^Anexo\s*\d*[A-Za-z]?", title, re.IGNORECASE):
                        is_valid_section = True

                    # c) Title Case corto (máx 4 palabras), sin ":" al final
                    elif (
                        len(title.split()) <= 4
                        and not title.endswith(":")
                        and re.match(r"^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]*$", title)
                    ):
                        is_valid_section = True

                    if not is_valid_section:
                        return None, None, 0

                # Si Docling SÍ lo marcó como header, confiar (sin validación de keywords)
                title = normalize_section_title(title)

            return match.group(1), title, level

    # Si no tiene número pero Docling lo marcó como header, intentar extraerlo
    if is_header_label and len(text) < 100:
        # Header sin número (ej: "INTRODUCCIÓN", "ANEXO")
        if text.isupper() or re.match(
            r"^[A-ZÁÉÍÓÚ][a-záéíóú]+(\s+[A-Za-záéíóú]+)*$", text
        ):
            return None, text, 1

    return None, None, 0


def infer_page_number(raw_text: str) -> Optional[int]:
    """Infiere el número de página a partir de patrones en el texto."""
    if not raw_text:
        return None
    m = re.search(r"\bHOJA\s*(\d{1,3})\s*de\s*\d{1,3}\b", raw_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bP[áa]gina\s*(\d{1,3})\s*de\s*\d{1,3}\b", raw_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


# =============================================================================
# NAVEGACIÓN DE JERARQUÍA
# =============================================================================


def get_parent_section(section_number: str) -> str:
    """Obtiene el número de sección padre."""
    if not section_number:
        return ""
    if section_number.lower().startswith("anexo"):
        return ""  # Anexo es raíz
    parts = section_number.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def get_root_section(section_number: str) -> str:
    """Obtiene el número de sección raíz (primer nivel)."""
    if not section_number:
        return ""
    if section_number.lower().startswith("anexo"):
        return section_number
    parts = section_number.split(".")
    return parts[0] if parts else ""


def get_subsection(section_number: str) -> str:
    """Obtiene la subsección (X.Y) de un número de sección."""
    if not section_number:
        return ""
    if section_number.lower().startswith("anexo"):
        return section_number
    parts = section_number.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return ""


# =============================================================================
# DETECCIÓN DE CONTENIDO ESPECIAL
# =============================================================================


def is_toc_line(text: str) -> bool:
    """Detecta si una línea es parte de una tabla de contenidos."""
    text = text.strip()
    if re.match(r"^[\d\.]+\s+[A-ZÁÉÍÓÚ].*\.{3,}\s*\d{1,3}\s*$", text):
        return True
    if re.match(r"^[\d\.]+\s+[A-ZÁÉÍÓÚ].*\s+\d{1,2}\s*$", text):
        return True
    section_numbers = re.findall(r"\b(\d+\.\d+|\d+\.)\s+[A-ZÁÉÍÓÚ]", text)
    if len(section_numbers) >= 2:
        return True
    return False


def is_index_content(text: str) -> bool:
    """Detecta si el texto es contenido de índice."""
    if re.match(r"^[1-9]\.\s+[A-ZÁÉÍÓÚ]{3,}", text.strip()) and not re.search(
        r"\.\.\.s*\d+\s*$", text
    ):
        return False

    lines = text.strip().split("\n")
    toc_lines = sum(1 for line in lines if is_toc_line(line))
    if len(lines) > 3 and toc_lines / len(lines) > 0.5:
        return True
    if re.search(r"\.{5,}\s*\d+", text):
        return True
    if "ÍNDICE" in text.upper() or "INDICE" in text.upper():
        return toc_lines > 2
    return False


def is_metadata_content(content: str, config: ChunkingConfig) -> bool:
    """Detecta si el contenido es metadata del documento (no contenido útil)."""
    content_lower = content.lower()

    # Excepciones para tablas útiles
    if re.search(
        r"\|.*(ciudad|estado|hotel|brand|property|chain|monto|nacional|extranjero|tabulador).*\|",
        content_lower,
    ):
        return False

    if re.search(r"\|.*hoja.*\|.*nd-a&f.*\|", content_lower):
        return True
    if re.search(r"\|.*fecha de creaci[oó]n.*\|", content_lower):
        return True

    if "|" in content_lower:
        if (
            "propietario del documento" in content_lower
            and "fechas de la revisión" in content_lower
        ):
            return True
        if (
            "descripción" in content_lower
            and "propósito" in content_lower
            and "autoriz" in content_lower
        ):
            return True

    return sum(1 for kw in config.metadata_keywords if kw in content_lower) >= 2


def is_ocr_garbage(text: str) -> bool:
    """Detecta si el texto es basura generada por OCR."""
    if len(text) < 10:
        return False
    if re.search(r"\|.*\|", text):
        return False

    consonant_clusters = re.findall(r"[bcdfghjklmnpqrstvwxz]{4,}", text.lower())
    if len(consonant_clusters) > 2:
        return True

    words = text.split()
    no_vowel_words = sum(
        1 for w in words if len(w) > 3 and not re.search(r"[aeiouáéíóú]", w.lower())
    )
    if len(words) > 3 and no_vowel_words / len(words) > 0.5:
        return True

    return False


def is_empty_table(text: str) -> bool:
    """Detecta si una tabla está esencialmente vacía."""
    if not re.search(r"\|.*\|", text):
        return False
    lines = [line.strip() for line in text.split("\n") if "|" in line]
    if not lines:
        return False

    empty_cells = 0
    total_cells = 0
    for line in lines:
        cells = [
            c.strip()
            for c in line.split("|")
            if c.strip() not in ["", "-", "---", "----"]
        ]
        total_cells += len(cells)
        empty_cells += sum(1 for c in cells if not c or c in ["-", "---", "----", ":"])

    if total_cells > 0 and empty_cells / total_cells > 0.7:
        return True
    if re.search(r"\|\s*[A-Za-z]+:\s*\|\s*\|", text):
        return True
    return False


# =============================================================================
# LIMPIEZA DE TOC
# =============================================================================


def clean_toc_bleed(text: str) -> str:
    """Limpia el sangrado de TOC que puede aparecer al inicio del contenido."""
    toc_pattern = re.search(
        r"^.*?([89]\.\d*\s+[A-ZÁÉÍÓÚ].*?)(1\.\s+[A-ZÁÉÍÓÚ]{3,})", text, re.DOTALL
    )
    if toc_pattern:
        prefix = toc_pattern.group(1)
        if "SISTEMA DE CONSECUENCIAS" in prefix or re.search(
            r"[89]\.\d+\s+[A-ZÁÉÍÓÚ]", prefix
        ):
            return text[toc_pattern.start(2) :]
    return text


__all__ = [
    "clean_text",
    "is_likely_list_item",
    "is_likely_section_title_line",
    "normalize_section_title",
    "extract_section_info",
    "infer_page_number",
    "get_parent_section",
    "get_root_section",
    "get_subsection",
    "is_toc_line",
    "is_index_content",
    "is_metadata_content",
    "is_ocr_garbage",
    "is_empty_table",
    "clean_toc_bleed",
]
