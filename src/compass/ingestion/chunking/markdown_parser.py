"""
Parser de Markdown estructurado a AST de secciones.
Convierte el output de Docling/Marker en un árbol jerárquico de secciones.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Section:
    """Representa una sección del documento con su jerarquía."""

    number: str  # "5.1" o "Anexo 1"
    title: str  # "ASIGNACIÓN"
    level: int  # Nivel jerárquico (1, 2, 3...)
    content: str  # Contenido Markdown de la sección
    raw_header: str  # Header original ("## 5.1 ASIGNACIÓN")
    children: List["Section"] = field(default_factory=list)
    parent: Optional["Section"] = None
    page_start: Optional[int] = None

    @property
    def full_path(self) -> str:
        """Construye el path completo: '5. NORMAS GENERALES > 5.1 ASIGNACIÓN'."""
        if self.parent:
            parent_path = self.parent.full_path
            self_label = f"{self.number}. {self.title}" if self.title else self.number
            return f"{parent_path} > {self_label}"
        return f"{self.number}. {self.title}" if self.title else self.number

    @property
    def root_section(self) -> "Section":
        """Obtiene la sección raíz de la jerarquía."""
        if self.parent:
            return self.parent.root_section
        return self

    def get_context_summary(self, max_chars: int = 200) -> str:
        """Obtiene un resumen del contexto padre para embedding."""
        if not self.parent:
            return ""
        parent_content = self.parent.content[:max_chars].strip()
        if len(self.parent.content) > max_chars:
            parent_content += "..."
        return f"[Contexto: {self.parent.full_path}] {parent_content}"


@dataclass
class ParsedDocument:
    """Documento parseado con todas sus secciones."""

    source: str  # Nombre del archivo fuente
    sections: List[Section] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def get_all_sections_flat(self) -> List[Section]:
        """Retorna todas las secciones en orden plano (pre-order traversal)."""
        result = []

        def traverse(section: Section):
            result.append(section)
            for child in section.children:
                traverse(child)

        for section in self.sections:
            traverse(section)
        return result

    def get_sections_at_level(self, level: int) -> List[Section]:
        """Retorna todas las secciones de un nivel específico."""
        return [s for s in self.get_all_sections_flat() if s.level == level]


class MarkdownSectionParser:
    """Parser que convierte Markdown estructurado a un árbol de secciones."""

    # Patrones para detectar headers y secciones
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s*\*{0,2}(.*?)\*{0,2}\s*$", re.MULTILINE)

    # Patrón para secciones numeradas: "1.", "1.1", "5.1.2", "Anexo 1"
    # Grupo 1: formato con subsecciones (1.1, 5.1.2)
    # Grupo 2: formato simple con punto (1., 5.)
    # Grupo 3: formato Anexo
    # Grupo 4: título restante
    SECTION_NUMBER_PATTERN = re.compile(
        r"^(?:(\d+(?:\.\d+)+)\.?\s*|(\d+)\.?\s+|(Anexo\s*\d*[A-Za-z]?)[\s:]*)(.*)$",
        re.IGNORECASE,
    )

    # Patrón estricto para promover líneas sin '#' a headers (ej: "4.1 Título")
    # Soporta negritas/itálicas envolviendo la línea completa.
    STRICT_SECTION_HEADER_PATTERN = re.compile(
        r"^\*{0,2}(\d+\.\d+(?:\.\d+)*)(?:\.)?\s+(.+?)\*{0,2}$"
    )

    # Headers no estructurales (no deben elevarse a sección)
    NON_STRUCTURAL_HEADER_PATTERN = re.compile(
        r"^(tabla|cuadro|figura|ilustraci[oó]n|gr[aá]fico)\b",
        re.IGNORECASE,
    )

    # Patrón para marcadores de página de Marker: {0}---
    PAGE_MARKER_PATTERN = re.compile(r"\{(\d+)\}[-─]{3,}")

    # Patrones de ruido a filtrar (líneas individuales)
    NOISE_PATTERNS = [
        re.compile(r"<!-- image -->", re.IGNORECASE),
        re.compile(r"!\[.*?\]\([^)]+\)"),  # Image references
    ]

    # Headers repetitivos de encabezado de página que deben ignorarse
    # Estos aparecen después de cada tabla de metadatos de página y rompen
    # la continuidad de las secciones (ej: rompen 5.1 creando múltiples chunks)
    PAGE_HEADER_NOISE = [
        "Spin",
        "Política Gastos de Viaje",
        "Política de Equipos de Cómputo",
        "Política de Eventos",
        "Digital@FEMSA",
        # Añadir títulos de documentos comunes aquí
    ]

    # Patrones para detectar tablas de metadatos completas (Digital@FEMSA, Spin, etc.)
    # Estas tablas suelen contener: HOJA, Código, Versión, Fecha de creación, etc.
    METADATA_TABLE_INDICATORS = [
        "HOJA",
        "Código",
        "Versión:",
        "Fecha de creación",
        "Fecha de modificación",
        "Estatus",
        "ND-",  # Códigos de documento como ND-A&F-EV-001
    ]

    # Indicadores de Tabla de Contenido (TOC) que deben excluirse de chunks
    TOC_INDICATORS = [
        "CONTENIDO",
        "ÍNDICE",
        "INDICE",
        "TABLA DE CONTENIDO",
        "TABLE OF CONTENTS",
    ]

    def __init__(
        self,
        filter_noise: bool = True,
        filter_metadata_tables: bool = True,
        filter_toc: bool = True,
    ):
        self.filter_noise = filter_noise
        self.filter_metadata_tables = filter_metadata_tables
        self.filter_toc = filter_toc

    def parse(
        self, markdown: str, source_name: str = "", page_map: dict = None
    ) -> ParsedDocument:
        """
        Parsea el Markdown y retorna un ParsedDocument con el árbol de secciones.

        Args:
            markdown: Contenido Markdown del documento
            source_name: Nombre del archivo fuente
            page_map: Mapa de páginas extraído de Docling (opcional)
                      {"items": [(text, page_no), ...], "headers": {text: page_no}}

        Returns:
            ParsedDocument con las secciones parseadas
        """
        # 1. Pre-procesar: extraer marcadores de página y limpiar ruido
        markdown, page_markers = self._extract_page_markers(markdown)

        if self.filter_noise:
            markdown = self._remove_noise(markdown)

        # 2. Dividir en bloques por headers
        blocks = self._split_by_headers(markdown)

        # 3. Construir secciones
        sections = self._build_sections(blocks, page_markers)

        # 4. Si tenemos page_map de Docling, usarlo para asignar páginas precisas
        if page_map:
            self._assign_pages_from_docling(sections, page_map)

        # 5. Construir jerarquía
        root_sections = self._build_hierarchy(sections)

        # Calcular page_count desde page_map si está disponible
        page_count = 0
        if page_map and page_map.get("items"):
            page_count = max(p for _, p in page_map["items"])
        elif page_markers:
            page_count = max(page_markers.values())

        return ParsedDocument(
            source=source_name,
            sections=root_sections,
            metadata={"page_count": page_count},
        )

    def _assign_pages_from_docling(
        self, sections: List[Section], page_map: dict
    ) -> None:
        """
        Asigna números de página a las secciones usando el mapa de Docling.

        Detecta page_start (página donde inicia la sección).

        Args:
            sections: Lista de secciones a actualizar (modificadas in-place)
            page_map: Mapa de páginas {"items": [(text, page_no), ...], "headers": {...}}
        """
        import re

        items = page_map.get("items", [])
        headers = page_map.get("headers", {})

        def extract_phrases(text: str) -> list:
            """Extrae frases de 3 palabras para búsqueda."""
            clean = re.sub(r"[^\w\sáéíóúñü]", " ", text.lower())
            words = [w for w in clean.split() if len(w) > 3]
            phrases = []
            for i in range(len(words) - 2):
                phrase = " ".join(words[i : i + 3])
                if len(phrase) > 12:
                    phrases.append(phrase)
            return phrases[:10]

        for section in sections:
            page_start = None

            # 1. Intentar match del título en headers
            title_key = section.title[:200] if section.title else ""
            for header_text, page_no in headers.items():
                if title_key and (title_key in header_text or header_text in title_key):
                    page_start = page_no
                    break

            # 2. Si no hay match de header, buscar por contenido inicial
            if page_start is None and section.content:
                content_clean = re.sub(r"[\*\#\|\-\[\]\.]", " ", section.content)
                content_clean = re.sub(r"\s+", " ", content_clean).strip()
                phrases = extract_phrases(content_clean[:400])

                for text_fragment, page_no in items:
                    clean_fragment = re.sub(
                        r"[^\w\sáéíóúñü]", " ", text_fragment.lower()
                    )
                    for phrase in phrases:
                        if phrase in clean_fragment:
                            page_start = page_no
                            break
                    if page_start:
                        break

            # 3. Asignar página
            if page_start is not None:
                section.page_start = page_start

    def _extract_page_markers(self, markdown: str) -> Tuple[str, Dict[int, int]]:
        """Extrae marcadores de página y retorna el markdown limpio + mapa de posiciones."""
        page_markers = {}

        def replace_marker(match):
            page_num = int(match.group(1))
            pos = match.start()
            page_markers[pos] = page_num
            return f"\n<!-- PAGE:{page_num} -->\n"

        cleaned = self.PAGE_MARKER_PATTERN.sub(replace_marker, markdown)
        return cleaned, page_markers

    def _remove_noise(self, markdown: str) -> str:
        """Elimina tablas de metadatos y ruido común."""
        result = markdown

        # 1. Aplicar patrones simples de ruido
        for pattern in self.NOISE_PATTERNS:
            result = pattern.sub("", result)

        # 2. Eliminar tablas de metadatos completas
        if self.filter_metadata_tables:
            result = self._remove_metadata_tables(result)

        # 3. Limpiar líneas vacías excesivas
        result = re.sub(r"\n{4,}", "\n\n\n", result)

        return result.strip()

    def _remove_metadata_tables(self, markdown: str) -> str:
        """
        Detecta y elimina tablas de metadatos completas.

        Una tabla de metadatos se identifica cuando contiene palabras clave
        como HOJA, Código, Versión, Fecha de creación, etc.
        """
        lines = markdown.split("\n")
        result_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Detectar inicio de tabla Markdown (línea con |)
            if "|" in line and self._is_table_row(line):
                # Capturar toda la tabla
                table_lines = [line]
                j = i + 1

                while j < len(lines) and self._is_table_row(lines[j]):
                    table_lines.append(lines[j])
                    j += 1

                # Verificar si es una tabla de metadatos
                table_text = "\n".join(table_lines)
                if not self._is_metadata_table(table_text):
                    # No es metadata, conservar
                    result_lines.extend(table_lines)
                # Si es metadata, no agregar (eliminar)

                i = j
            else:
                result_lines.append(line)
                i += 1

        return "\n".join(result_lines)

    def _is_table_row(self, line: str) -> bool:
        """Verifica si una línea es parte de una tabla Markdown."""
        stripped = line.strip()
        if not stripped:
            return False
        # Tabla Markdown: empieza y termina con | o es separador (|---|---|)
        return stripped.startswith("|") or ("|" in stripped and "-" in stripped)

    def _is_metadata_table(self, table_text: str) -> bool:
        """
        Determina si una tabla contiene metadatos de documento.

        Busca indicadores como HOJA, Código, Versión, etc.
        """
        text_upper = table_text.upper()
        matches = sum(
            1
            for indicator in self.METADATA_TABLE_INDICATORS
            if indicator.upper() in text_upper
        )
        # Si tiene 2+ indicadores, es tabla de metadatos
        return matches >= 2

    def _split_by_headers(self, markdown: str) -> List[Dict]:
        """Divide el markdown en bloques basados en headers."""
        blocks = []
        lines = markdown.split("\n")
        current_block = {
            "header": None,
            "header_level": 0,
            "content": [],
            "line_start": 0,
        }

        for i, line in enumerate(lines):
            header_match = self.HEADER_PATTERN.match(line)

            if header_match:
                # Guardar bloque anterior si tiene contenido
                if current_block["content"] or current_block["header"]:
                    current_block["content"] = "\n".join(
                        current_block["content"]
                    ).strip()
                    blocks.append(current_block)

                # Iniciar nuevo bloque
                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                current_block = {
                    "header": header_text,
                    "header_level": header_level,
                    "raw_header": line,
                    "content": [],
                    "line_start": i,
                }
            else:
                strict_match = self.STRICT_SECTION_HEADER_PATTERN.match(line.strip())
                if strict_match:
                    # Promover a header con nivel basado en numeración
                    section_num = strict_match.group(1)
                    header_level = len(section_num.split("."))
                    header_text = f"{section_num} {strict_match.group(2).strip()}"

                    if current_block["content"] or current_block["header"]:
                        current_block["content"] = "\n".join(
                            current_block["content"]
                        ).strip()
                        blocks.append(current_block)

                    current_block = {
                        "header": header_text,
                        "header_level": header_level,
                        "raw_header": line.strip(),
                        "content": [],
                        "line_start": i,
                    }
                    continue
                current_block["content"].append(line)

        # Agregar último bloque
        if current_block["content"] or current_block["header"]:
            current_block["content"] = "\n".join(current_block["content"]).strip()
            blocks.append(current_block)

        return blocks

    def _build_sections(
        self, blocks: List[Dict], page_markers: Dict[int, int]
    ) -> List[Section]:
        """
        Construye objetos Section a partir de los bloques.

        Cuando un header es rechazado como falso positivo (ej: "1. Por robo:"),
        su contenido se fusiona con la sección anterior para no perder información.
        """
        sections = []
        pending_content = []  # Contenido de headers rechazados para fusionar

        for block in blocks:
            header = block.get("header", "")
            content = block.get("content", "")

            if not header:
                # Bloque sin header - acumular contenido
                if content:
                    pending_content.append(content)
                continue

            # Extraer número y título de sección
            section_num, section_title, level = self._extract_section_info(header)

            if not section_num and not section_title:
                # Header rechazado (falso positivo como "1. Por robo:")
                # Preservar el header original + contenido para fusionar después
                rejected_text = (
                    f"**{header}**\n{content}" if content else f"**{header}**"
                )
                pending_content.append(rejected_text)
                continue

            # Filtrar secciones de Tabla de Contenido (TOC)
            if self.filter_toc and self._is_toc_section(
                section_title or header, content
            ):
                # Skip TOC sections completely - don't add to pending_content
                continue

            # Header válido - crear sección
            # Primero, fusionar cualquier contenido pendiente con la sección anterior
            if pending_content and sections:
                sections[-1].content += "\n\n" + "\n\n".join(pending_content)
                pending_content = []

            # Estimar página basado en posición
            page_start = self._estimate_page(block.get("line_start", 0), page_markers)

            section = Section(
                number=section_num or "",
                title=section_title or header,
                level=level or block.get("header_level", 1),
                content=content,
                raw_header=block.get("raw_header", header),
                page_start=page_start,
            )
            sections.append(section)

        # Fusionar cualquier contenido pendiente final con la última sección
        if pending_content and sections:
            sections[-1].content += "\n\n" + "\n\n".join(pending_content)

        return sections

    def _is_toc_section(self, title: str, content: str) -> bool:
        """
        Detecta si una sección es una Tabla de Contenido (TOC).

        Una sección es TOC si:
        1. Su título contiene indicadores de TOC (CONTENIDO, ÍNDICE, etc.)
        2. Su contenido tiene estructura de TOC (líneas con .... y números de página)

        Args:
            title: Título de la sección
            content: Contenido de la sección

        Returns:
            True si la sección es una TOC que debe excluirse
        """
        # 1. Verificar título
        title_upper = title.upper().strip()
        for indicator in self.TOC_INDICATORS:
            if indicator in title_upper:
                return True

        # 2. Verificar estructura de contenido (líneas con .... y números)
        # Patrón: texto seguido de puntos y número de página
        toc_line_patterns = [
            r"\.{3,}\s*\d+",  # "Introducción ........... 4"
            r"→\s*\d+",  # "Introducción → 4"
            r"\t+\d+$",  # "Introducción\t\t4"
            r"\s{5,}\d+$",  # "Introducción     4" (muchos espacios)
        ]
        toc_matches = 0
        for pattern in toc_line_patterns:
            toc_matches += len(re.findall(pattern, content, re.MULTILINE))

        # Si hay 3+ líneas con patrón TOC, es una tabla de contenido
        if toc_matches >= 3:
            return True

        return False

    def _extract_section_info(
        self, header: str
    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        Extrae número de sección, título y nivel del header.

        Filtra falsos positivos como items de lista ("2. Daño material:")
        que terminan en ":" o tienen formato de item.

        También filtra headers repetitivos de encabezado de página como
        "Spin", "Política Gastos de Viaje" que aparecen después de cada
        tabla de metadatos y rompen la continuidad de las secciones.

        Returns:
            (section_number, section_title, hierarchy_level)
        """
        header = header.strip()
        header = re.sub(r"^\*{1,2}(.+?)\*{1,2}$", r"\1", header).strip()

        # FILTRO: Headers no estructurales (Tabla, Figura, etc.)
        if self.NON_STRUCTURAL_HEADER_PATTERN.match(header):
            return None, None, 0

        # FILTRO: Headers repetitivos de encabezado de página (ruido)
        # Estos aparecen después de tablas de metadata y rompen la jerarquía
        header_normalized = header.lower().strip()
        for noise_header in self.PAGE_HEADER_NOISE:
            if header_normalized == noise_header.lower():
                return None, None, 0

        # FILTRO: Headers que terminan en ":" son items de lista, NO secciones
        # Ej: "1. Por robo:", "2. Daño material:" → items de lista
        if header.endswith(":"):
            return None, None, 0

        # FILTRO: Headers que empiezan con minúscula después del número → items
        # Ej: "1. el colaborador debe..." → item de lista
        if re.match(r"^\d+\.?\s+[a-záéíóúñ]", header):
            return None, None, 0

        match = self.SECTION_NUMBER_PATTERN.match(header)
        if match:
            # Grupo 1: "1.1.2" formato con puntos
            # Grupo 2: "1" formato simple
            # Grupo 3: "Anexo 1" formato anexo
            # Grupo 4: título restante

            if match.group(1):  # Formato X.Y.Z
                section_num = match.group(1)
                level = len(section_num.split("."))
            elif match.group(2):  # Formato X
                section_num = match.group(2)
                level = 1
            elif match.group(3):  # Formato Anexo
                section_num = match.group(3).strip()
                level = 1
            else:
                return None, header, 1

            title = match.group(4).strip() if match.group(4) else ""

            # FILTRO adicional: Si el título extraído termina en ":" → item
            if title.endswith(":"):
                return None, None, 0

            # FILTRO: Título muy largo o con estructura de oración → no es sección
            # Permitir títulos numerados más largos (políticas suelen ser extensas).
            title_word_count = len(title.split())
            if title_word_count > 8:
                if not section_num or title_word_count > 16:
                    return None, None, 0

            return section_num, title, level

        # Header sin número (solo título)
        if header.isupper() and len(header) > 3:
            return None, header, 1

        return None, header, 1

    def _estimate_page(
        self, line_number: int, page_markers: Dict[int, int]
    ) -> Optional[int]:
        """Estima el número de página basado en marcadores."""
        if not page_markers:
            return None

        # Encontrar el marcador más cercano anterior a esta línea
        applicable_markers = [
            (pos, page) for pos, page in page_markers.items() if pos <= line_number
        ]
        if applicable_markers:
            return max(applicable_markers, key=lambda x: x[0])[1]
        return 1

    def _build_hierarchy(self, sections: List[Section]) -> List[Section]:
        """Construye la jerarquía padre-hijo entre secciones."""
        if not sections:
            return []

        root_sections = []
        section_stack: List[Section] = []

        for section in sections:
            # Encontrar el padre correcto basado en la numeración
            parent = self._find_parent(section, section_stack)

            if parent:
                section.parent = parent
                parent.children.append(section)
            else:
                root_sections.append(section)

            # Actualizar el stack
            # Remover secciones del stack que no pueden ser padres de futuras secciones
            while section_stack and not self._can_be_parent(section_stack[-1], section):
                section_stack.pop()

            section_stack.append(section)

        return root_sections

    def _find_parent(self, section: Section, stack: List[Section]) -> Optional[Section]:
        """Encuentra el padre de una sección basado en su numeración."""
        if not section.number or not stack:
            return None

        section_parts = section.number.split(".")

        if len(section_parts) <= 1:
            return None  # Secciones de nivel 1 no tienen padre

        # Buscar padre por número (ej: para "5.1" buscar "5")
        parent_number = ".".join(section_parts[:-1])

        for s in reversed(stack):
            if s.number == parent_number:
                return s

        # Fallback: buscar sección de nivel superior más reciente
        for s in reversed(stack):
            if s.level < section.level:
                return s

        return None

    def _can_be_parent(self, potential_parent: Section, child: Section) -> bool:
        """Determina si una sección puede ser padre de otra."""
        if not child.number:
            return potential_parent.level < child.level

        child_parts = child.number.split(".")
        if len(child_parts) <= 1:
            return False

        parent_prefix = ".".join(child_parts[:-1])
        return (
            potential_parent.number == parent_prefix
            or potential_parent.level < child.level
        )


def parse_docling_markdown(doc, source_name: str = "") -> ParsedDocument:
    """
    Convierte un documento de Docling a ParsedDocument.

    Args:
        doc: Documento de Docling (result.document)
        source_name: Nombre del archivo fuente

    Returns:
        ParsedDocument con las secciones parseadas
    """
    try:
        markdown = doc.export_to_markdown()
    except Exception as e:
        raise ValueError(f"Error exportando documento a Markdown: {e}")

    parser = MarkdownSectionParser()
    return parser.parse(markdown, source_name)


__all__ = [
    "Section",
    "ParsedDocument",
    "MarkdownSectionParser",
    "parse_docling_markdown",
]
