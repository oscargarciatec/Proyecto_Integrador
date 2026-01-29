"""
Estrategia de chunking para documentos de políticas corporativas.

Este módulo implementa PolicyDocumentChunker, optimizado para documentos
jerárquicos como políticas de RH, finanzas, IT, etc.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from .markdown_parser import Section, ParsedDocument
from .settings import ChunkingConfig, count_tokens, tokenizer


@dataclass
class Chunk:
    """Representa un chunk de texto listo para embedding."""

    id: str
    content: str  # Contenido principal del chunk
    embedding_content: str  # Contenido optimizado para embedding
    section_path: str  # "5. NORMAS > 5.1 ASIGNACIÓN"
    section_title: str  # Título de la sección raíz
    section_number: str  # "5.1"
    hierarchy_level: int  # Nivel en la jerarquía
    page_start: Optional[int] = None
    token_count: int = 0
    char_count: int = 0
    has_table: bool = False
    has_list: bool = False
    has_overlap: bool = False
    source_pdf: str = ""
    parent_context: str = ""  # Contexto del padre para retrieval

    def to_dict(self) -> Dict:
        """Convierte el chunk a diccionario para serialización."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding_content": self.embedding_content,
            "section_path": self.section_path,
            "section_title": self.section_title,
            "section_number": self.section_number,
            "hierarchy_level": self.hierarchy_level,
            "page_start": self.page_start,
            "token_count": self.token_count,
            "char_count": len(self.content),
            "block_count": 1,
            "block_types": self._infer_block_types(),
            "has_table": self.has_table,
            "has_overlap": self.has_overlap,
            "source_pdf": self.source_pdf,
        }

    def _infer_block_types(self) -> List[str]:
        """Infiere los tipos de bloques presentes en el contenido."""
        types = ["Text"]
        if self.has_table or "|" in self.content:
            types.append("Table")
        if self.has_list or re.search(r"^\s*[-*]\s", self.content, re.MULTILINE):
            types.append("List")
        return types


class PolicyDocumentChunker:
    """
    Estrategia de chunking optimizada para documentos de políticas corporativas.

    Características:
    - Fusiona secciones pequeñas con la siguiente sección grande
    - Incluye contexto padre en el embedding para mejor retrieval
    - Aplica overlap entre chunks consecutivos
    - Marca secciones fusionadas con headers bold para claridad
    """

    def __init__(self, config: ChunkingConfig = None, parent_context_tokens: int = 150):
        self.config = config or ChunkingConfig()
        self.parent_context_tokens = parent_context_tokens

    def create_chunks(
        self, document: ParsedDocument, config: ChunkingConfig = None
    ) -> List[Chunk]:
        """Crea chunks a partir de un documento parseado."""
        cfg = config or self.config
        chunks = []

        sections = document.get_all_sections_flat()

        # Acumular secciones pequeñas para fusionarlas con la siguiente
        pending_small_sections: List[tuple] = []

        for section in sections:
            content = self._clean_content(section.content, cfg)
            content_tokens = count_tokens(content)

            if content_tokens < cfg.min_tokens:
                # Guardar esta sección pequeña para fusionarla después
                if content.strip():
                    section_header = (
                        f"**{section.full_path}**\n" if section.full_path else ""
                    )
                    pending_small_sections.append((section, section_header + content))
                continue

            # Tenemos una sección lo suficientemente grande
            # Prepend cualquier sección pequeña pendiente
            if pending_small_sections:
                prepended_content = "\n\n".join(
                    [ps[1] for ps in pending_small_sections]
                )
                current_section_header = (
                    f"**{section.full_path}**\n" if section.full_path else ""
                )
                content = prepended_content + "\n\n" + current_section_header + content
                pending_small_sections = []

            features = self._detect_content_features(content)

            # Obtener contexto del padre
            parent_context = ""
            if section.parent:
                parent_context = self._get_parent_summary(section.parent)

            # Dividir si es muy grande
            content_parts = self._split_large_content(content, cfg.max_tokens)

            for part_idx, part in enumerate(content_parts):
                chunk_id = f"chunk_{len(chunks):03d}"
                suffix = (
                    f" (Parte {part_idx + 1}/{len(content_parts)})"
                    if len(content_parts) > 1
                    else ""
                )

                embedding = self._build_embedding_content(
                    document.source, section.full_path + suffix, part, parent_context
                )

                chunk = Chunk(
                    id=chunk_id,
                    content=part,
                    embedding_content=embedding,
                    section_path=section.full_path,
                    section_title=section.root_section.title,
                    section_number=section.number,
                    hierarchy_level=section.level,
                    page_start=section.page_start,
                    token_count=count_tokens(part),
                    source_pdf=document.source,
                    parent_context=parent_context,
                    **features,
                )
                chunks.append(chunk)

        # Si quedaron secciones pequeñas al final, agregarlas como chunk
        if pending_small_sections:
            combined_content = "\n\n".join([ps[1] for ps in pending_small_sections])
            if count_tokens(combined_content) >= cfg.min_tokens // 2:
                last_section = pending_small_sections[-1][0]
                features = self._detect_content_features(combined_content)
                chunk = Chunk(
                    id=f"chunk_{len(chunks):03d}",
                    content=combined_content,
                    embedding_content=self._build_embedding_content(
                        document.source, "Secciones Adicionales", combined_content, ""
                    ),
                    section_path="Secciones Adicionales",
                    section_title=last_section.title if last_section else "",
                    section_number="",
                    hierarchy_level=1,
                    page_start=None,
                    token_count=count_tokens(combined_content),
                    source_pdf=document.source,
                    parent_context="",
                    **features,
                )
                chunks.append(chunk)

        # Aplicar overlap entre chunks consecutivos
        if cfg.overlap_tokens > 0:
            chunks = self._apply_overlap(chunks, cfg.overlap_tokens)

        return chunks

    def _clean_content(self, content: str, cfg: ChunkingConfig) -> str:
        """Limpia el contenido aplicando patrones de filtro."""
        cleaned = content
        for pattern in cfg.filter_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r" +", " ", cleaned)
        return cleaned.strip()

    def _detect_content_features(self, content: str) -> Dict[str, bool]:
        """Detecta características del contenido (tablas, listas, etc.)."""
        return {
            "has_table": bool(re.search(r"\|.*\|", content)),
            "has_list": bool(re.search(r"^\s*[-*•]\s", content, re.MULTILINE)),
        }

    def _get_parent_summary(self, parent: Section) -> str:
        """Obtiene un resumen del contexto padre."""
        parent_content = parent.content
        tokens = tokenizer.encode(parent_content)

        if len(tokens) > self.parent_context_tokens:
            parent_content = (
                tokenizer.decode(tokens[: self.parent_context_tokens]) + "..."
            )

        return f"{parent.full_path}: {parent_content[:200]}"

    def _build_embedding_content(
        self, source: str, section_path: str, content: str, parent_context: str
    ) -> str:
        """Construye embedding content con contexto padre integrado."""
        parts = [source, section_path]
        if parent_context:
            parts.append(f"[Contexto: {parent_context}]")
        parts.append(content)
        return "-".join(filter(None, parts))

    def _split_large_content(
        self, content: str, max_tokens: int, separators: List[str] = None
    ) -> List[str]:
        """Divide contenido grande respetando límites de tokens."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]

        if count_tokens(content) <= max_tokens:
            return [content]

        if not separators:
            tokens = tokenizer.encode(content)
            parts = []
            for i in range(0, len(tokens), max_tokens):
                parts.append(tokenizer.decode(tokens[i : i + max_tokens]))
            return parts

        separator = separators[0]
        splits = content.split(separator) if separator else list(content)

        result = []
        current = []
        current_tokens = 0

        for split in splits:
            split_tokens = count_tokens(split)

            if split_tokens > max_tokens:
                if current:
                    result.append(separator.join(current))
                    current = []
                    current_tokens = 0
                result.extend(
                    self._split_large_content(split, max_tokens, separators[1:])
                )
            elif current_tokens + split_tokens + count_tokens(separator) > max_tokens:
                if current:
                    result.append(separator.join(current))
                current = [split]
                current_tokens = split_tokens
            else:
                current.append(split)
                current_tokens += split_tokens + (
                    count_tokens(separator) if current else 0
                )

        if current:
            result.append(separator.join(current))

        return result

    def _apply_overlap(self, chunks: List[Chunk], overlap_tokens: int) -> List[Chunk]:
        """Aplica overlap entre chunks consecutivos."""
        if len(chunks) <= 1:
            return chunks

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            prev_tokens = tokenizer.encode(prev_chunk.content)
            if len(prev_tokens) > overlap_tokens:
                overlap_tokens_list = prev_tokens[-overlap_tokens:]
                overlap_text = tokenizer.decode(overlap_tokens_list)

                last_sentence = overlap_text.find(". ")
                if last_sentence > len(overlap_text) // 3:
                    overlap_text = overlap_text[last_sentence + 2 :]

                curr_chunk.content = f"[...] {overlap_text}\n\n{curr_chunk.content}"
                curr_chunk.has_overlap = True
                curr_chunk.token_count = count_tokens(curr_chunk.content)

        return chunks


# Alias para compatibilidad con código existente
SmallToBigChunker = PolicyDocumentChunker


__all__ = [
    "Chunk",
    "PolicyDocumentChunker",
    "SmallToBigChunker",
]
