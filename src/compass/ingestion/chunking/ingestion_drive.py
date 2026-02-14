"""
Ingesta de chunks semánticos desde Google Drive a AlloyDB.

Lee JSON chunks desde la subcarpeta de Drive y los ingesta a AlloyDB.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras as extras
from loguru import logger

from google import genai
from google.genai.types import EmbedContentConfig


@dataclass
class IngestionConfig:
    """Configuración para la ingesta de chunks desde Drive."""

    # Drive
    drive_pdf_folder_id: str | None = None
    drive_chunks_subfolder: str = "chunks"

    # Vertex AI
    project_id: str = ""
    vertex_location: str = ""
    embedding_model: str = ""

    # AlloyDB
    alloydb_host: str | None = None
    alloydb_port: int = 5432
    alloydb_database: str | None = None
    alloydb_user: str | None = None
    alloydb_password: str | None = None
    alloydb_sslmode: str = "require"

    # Tables (configurable via env vars for test vs prod)
    knowledge_table: str = ""
    metadata_table: str = ""
    hub_knowledge_table: str = ""
    source_system: str = ""
    hub_src_system_id: int = 0

    def __post_init__(self):
        """Set defaults from environment if not provided."""
        import os

        if not self.project_id:
            self.project_id = os.getenv(
                "GCS_PROJECT_ID", os.getenv("GCP_PROJECT_ID", "")
            )
        if not self.vertex_location:
            self.vertex_location = os.getenv("VERTEX_LOCATION", "us-east1")
        if not self.embedding_model:
            self.embedding_model = os.getenv(
                "EMBEDDING_MODEL", "text-multilingual-embedding-002"
            )
        if not self.knowledge_table:
            self.knowledge_table = os.getenv(
                "KNOWLEDGE_TABLE", "multiagent_rag_model.sat_knowledge"
            )
        if not self.metadata_table:
            self.metadata_table = os.getenv(
                "METADATA_TABLE", "multiagent_rag_model.sat_knowledge_metadata"
            )
        if not self.hub_knowledge_table:
            self.hub_knowledge_table = os.getenv(
                "HUB_KNOWLEDGE_TABLE", "multiagent_rag_model.hub_knowledge"
            )
        if not self.source_system:
            self.source_system = os.getenv("SOURCE_SYSTEM", "gen-ai spin-compass")
        if self.hub_src_system_id == 0:
            self.hub_src_system_id = int(os.getenv("HUB_SRC_SYSTEM_ID", "31"))


@dataclass
class IngestionResult:
    """Resultado de la ingesta."""

    files_processed: int = 0
    chunks_inserted: int = 0
    chunks_updated: int = 0
    hub_inserted: int = 0
    errors: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "chunks_inserted": self.chunks_inserted,
            "chunks_updated": self.chunks_updated,
            "hub_inserted": self.hub_inserted,
            "errors": self.errors or [],
        }


class DriveChunkIngester:
    """
    Ingesta chunks semánticos desde Drive a AlloyDB.

    Flujo:
      1. Lista JSONs en la subcarpeta de Drive
      2. Para cada JSON:
         - Descarga y parsea los chunks
         - Genera embeddings con Vertex AI
         - Ingesta a AlloyDB (SCD2)
    """

    def __init__(self, config: IngestionConfig):
        self.config = config
        self._drive_service = None
        self._genai_client = None
        self._conn = None
        self._chunks_folder_id = None

    @property
    def drive_service(self):
        if self._drive_service is None:
            from ..sources.drive_service import get_drive_service

            self._drive_service = get_drive_service()
        return self._drive_service

    @property
    def genai_client(self) -> genai.Client:
        """Cliente de Google GenAI para embeddings."""
        if self._genai_client is None:
            self._genai_client = genai.Client(
                vertexai=True,
                project=self.config.project_id,
                location=self.config.vertex_location,
            )
            logger.info(
                f"✅ Cliente GenAI inicializado para '{self.config.embedding_model}'"
            )
        return self._genai_client

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                host=self.config.alloydb_host,
                port=self.config.alloydb_port,
                database=self.config.alloydb_database,
                user=self.config.alloydb_user,
                password=self.config.alloydb_password,
                sslmode=self.config.alloydb_sslmode,
            )
            logger.info("✅ Conexión a AlloyDB establecida")
        return self._conn

    def _get_chunks_folder_id(self) -> str:
        """Obtiene el ID de la subcarpeta de chunks."""
        if self._chunks_folder_id:
            return self._chunks_folder_id

        from ..sources.drive_service import drive_list_all

        parent_id = self.config.drive_pdf_folder_id
        target_name = self.config.drive_chunks_subfolder

        # Debug: mostrar información de la carpeta padre
        try:
            parent_info = (
                self.drive_service.files()
                .get(fileId=parent_id, fields="id, name", supportsAllDrives=True)
                .execute()
            )
            logger.warning(
                f"🔍 [DEBUG] Buscando '{target_name}' en carpeta: "
                f"ID={parent_info['id']}, Nombre='{parent_info['name']}'"
            )
        except Exception as e:
            logger.error(f"❌ Error obteniendo info de carpeta padre {parent_id}: {e}")

        query = (
            f"'{parent_id}' in parents "
            f"and name='{target_name}' "
            f"and mimeType='application/vnd.google-apps.folder' "
            f"and trashed=false"
        )

        folders = drive_list_all(
            self.drive_service,
            q=query,
            fields="files(id, name)",
        )

        if not folders:
            # Diagnóstico: listar qué subcarpetas SÍ existen
            all_subs_query = (
                f"'{parent_id}' in parents "
                f"and mimeType='application/vnd.google-apps.folder' "
                f"and trashed=false"
            )
            all_subs = drive_list_all(
                self.drive_service, q=all_subs_query, fields="files(id, name)"
            )
            sub_names = [s.get("name") for s in all_subs]
            logger.error(f"❌ Subcarpeta '{target_name}' NO encontrada")
            logger.error(f"   📂 Subcarpetas disponibles en {parent_id}: {sub_names}")

            raise ValueError(
                f"Subcarpeta '{target_name}' no encontrada en carpeta {parent_id}. "
                f"Subcarpetas disponibles: {sub_names}"
            )

        self._chunks_folder_id = folders[0].get("id")
        logger.info(
            f"✅ Subcarpeta '{target_name}' encontrada: {self._chunks_folder_id}"
        )
        return self._chunks_folder_id

    def run(self) -> dict:
        """Ejecuta la ingesta completa."""
        result = IngestionResult(errors=[])

        try:
            from ..sources.drive_service import drive_list_all

            chunks_folder_id = self._get_chunks_folder_id()

            # Listar JSONs en la subcarpeta (incluir md5Checksum y appProperties)
            query = f"'{chunks_folder_id}' in parents and trashed=false"
            all_items = drive_list_all(
                self.drive_service,
                q=query,
                fields="files(id, name, md5Checksum, appProperties)",
            )

            json_files = [f for f in all_items if f.get("name", "").endswith(".json")]
            logger.info(f"📂 Encontrados {len(json_files)} archivos JSON en chunks/")

            for file_info in json_files:
                try:
                    # Verificar si ya fue procesado (skip por md5)
                    if self._should_skip_json(file_info):
                        logger.debug(f"⏭️ Skip: {file_info.get('name')} ya procesado")
                        continue

                    self._process_drive_json(file_info, result)

                    # Marcar como procesado después de éxito
                    self._mark_json_as_processed(file_info)

                except Exception as e:
                    error_msg = f"{file_info.get('name')}: {e}"
                    logger.error(f"❌ {error_msg}")
                    result.errors.append(error_msg)

        finally:
            if self._conn:
                self._conn.close()
                logger.info("Conexión a AlloyDB cerrada")

        return result.to_dict()

    def _should_skip_json(self, file_info: dict) -> bool:
        """Verifica si el archivo JSON ya fue procesado (mismo md5)."""
        app_props = file_info.get("appProperties") or {}
        processed_md5 = app_props.get("ingested_md5")
        current_md5 = file_info.get("md5Checksum")

        # Si no hay md5 guardado, no se ha procesado
        if not processed_md5:
            return False

        # Si el md5 actual es igual al procesado, skip
        return processed_md5 == current_md5

    def _mark_json_as_processed(self, file_info: dict):
        """Marca el archivo JSON como procesado actualizando appProperties."""
        file_id = file_info.get("id")
        current_md5 = file_info.get("md5Checksum")
        app_props = file_info.get("appProperties") or {}

        new_props = dict(app_props)
        new_props["ingested_md5"] = str(current_md5 or "")
        new_props["ingested_at"] = datetime.now(timezone.utc).isoformat() + "Z"

        try:
            self.drive_service.files().update(
                fileId=file_id,
                body={"appProperties": new_props},
                fields="id",
                supportsAllDrives=True,
            ).execute()
        except Exception as e:
            logger.warning(f"⚠️ No se pudo marcar como procesado: {e}")

    def _process_drive_json(self, file_info: dict, result: IngestionResult):
        """Procesa un archivo JSON de chunks desde Drive."""
        file_id = file_info.get("id")
        file_name = file_info.get("name")

        logger.info(f"🚀 Procesando: {file_name}")

        # Descargar JSON
        request = self.drive_service.files().get_media(
            fileId=file_id, supportsAllDrives=True
        )
        content = request.execute()

        data = json.loads(content.decode("utf-8"))
        chunks = data.get("chunks", [])

        if not chunks:
            logger.warning(f"⚠️ Archivo sin chunks: {file_name}")
            return

        logger.info(f"   📦 {len(chunks)} chunks cargados")

        # Generar embeddings
        embeddings = self._get_embeddings_batch(chunks)
        for i, ch in enumerate(chunks):
            ch["embedding_vector"] = embeddings[i]

        # Preparar filas
        knowledge_rows, metadata_rows, hub_rows = self._prepare_rows(chunks)

        # Ingestar
        hub_result = self._ingest_hub_knowledge(hub_rows)
        k_result = self._ingest_knowledge(knowledge_rows)
        m_result = self._ingest_metadata(metadata_rows)

        result.files_processed += 1
        result.chunks_inserted += k_result.get("inserted", 0)
        result.chunks_updated += k_result.get("updated", 0)
        result.hub_inserted += hub_result.get("inserted", 0)

        logger.info(
            f"   ✅ Hub={hub_result}, Knowledge={k_result}, Metadata={m_result}"
        )

    def _get_embeddings_batch(self, chunks: list[dict]) -> list[list[float] | None]:
        """Genera embeddings en lotes."""
        api_limit = 50
        all_embeddings = []

        contents = [ch.get("embedding_content", ch.get("content", "")) for ch in chunks]

        for i in range(0, len(contents), api_limit):
            batch = contents[i : i + api_limit]
            batch_embeddings = [None] * len(batch)

            valid_items = [
                (idx, text) for idx, text in enumerate(batch) if text and text.strip()
            ]

            if not valid_items:
                all_embeddings.extend(batch_embeddings)
                continue

            valid_indices = [item[0] for item in valid_items]
            valid_texts = [item[1] for item in valid_items]

            logger.info(
                f"   → Embeddings para {len(valid_texts)} chunks (lote {i // api_limit + 1})"
            )

            try:
                response = self.genai_client.models.embed_content(
                    model=self.config.embedding_model,
                    contents=valid_texts,
                    config=EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                for orig_idx, emb in zip(valid_indices, response.embeddings):
                    batch_embeddings[orig_idx] = emb.values
            except Exception as e:
                logger.error(f"⚠️ Error embeddings: {e}")

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _prepare_rows(self, chunks: list[dict]) -> tuple[list, list, list]:
        """Prepara filas para las 3 tablas."""
        by_src: dict[str, list] = {}
        for ch in chunks:
            src = ch.get("source_pdf", "unknown")
            by_src.setdefault(src, []).append(ch)

        knowledge_rows = []
        metadata_rows = []
        hub_rows = []

        for src, items in by_src.items():
            kh_input = f"{src}|{self.config.source_system}"
            kh = _sha1_bytes(kh_input)
            ts_hub = _now_utc().replace(microsecond=0)

            hub_rows.append(
                {
                    "kh_knowledge": kh,
                    "ax_knowledge": src,
                    "ct_ingest_dt": ts_hub,
                    "ax_src_system_datastore": self.config.source_system,
                    "ai_src_system": self.config.hub_src_system_id,
                }
            )

            for ch in items:
                chunk_id = ch.get("id", "semantic_chunk_0")
                subseq = _build_subseq_semantic(chunk_id, kh)
                cs = _checksum_semantic(ch)

                content = ch.get("content", "")
                embedding_content = ch.get("embedding_content", content)
                ts = _now_utc().replace(microsecond=0)
                vec_string = _to_pgvector(ch.get("embedding_vector"))

                knowledge_rows.append(
                    {
                        "kh_knowledge": kh,
                        "ax_sub_sequence": subseq,
                        "ah_checksum": cs,
                        "ax_content": embedding_content,
                        "aa_embedding": vec_string,
                        "ct_valid_from_dt": ts,
                        "ct_ingest_dt": ts,
                        "ax_src_system_datastore": self.config.source_system,
                    }
                )

                section_title = ch.get("section_title", "")
                block_types = ch.get("block_types", [])
                chunk_type_json = json.dumps({"block_types": block_types})

                metadata_rows.append(
                    {
                        "kh_knowledge": kh,
                        "ax_sub_sequence": subseq,
                        "ah_checksum": cs,
                        "ai_page_number": ch.get("page_start", 0),
                        "ax_chunk_type": chunk_type_json,
                        "ax_content": embedding_content,
                        "ai_char_count": ch.get("char_count", 0),
                        "ai_token_count": ch.get("token_count", 0),
                        "ax_topic": section_title,
                        "ax_source": src,
                        "ct_valid_from_dt": ts,
                        "ct_ingest_dt": ts,
                        "ax_src_system_datastore": self.config.source_system,
                    }
                )

        return knowledge_rows, metadata_rows, hub_rows

    def _ingest_hub_knowledge(self, rows: list[dict]) -> dict:
        """Ingesta registros en hub_knowledge."""
        if not rows:
            return {"inserted": 0, "skipped": 0}

        with self.conn:
            with self.conn.cursor() as c:
                inserted = 0
                skipped = 0

                for r in rows:
                    c.execute(
                        f"SELECT 1 FROM {self.config.hub_knowledge_table} WHERE kh_knowledge = %s",
                        (r["kh_knowledge"],),
                    )
                    if c.fetchone():
                        skipped += 1
                        continue

                    c.execute(
                        f"""
                        INSERT INTO {self.config.hub_knowledge_table}
                        (kh_knowledge, ax_knowledge, ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            r["kh_knowledge"],
                            r["ax_knowledge"],
                            r["ct_ingest_dt"],
                            r["ax_src_system_datastore"],
                            r["ai_src_system"],
                        ),
                    )
                    inserted += 1

                return {"inserted": inserted, "skipped": skipped}

    def _ingest_knowledge(self, rows: list[dict]) -> dict:
        """Ingesta chunks con SCD2."""
        if not rows:
            return {"updated": 0, "inserted": 0}

        with self.conn:
            with self.conn.cursor() as c:
                c.execute("""
                    CREATE TEMP TABLE stage_k (
                        kh_knowledge BYTEA,
                        ax_sub_sequence VARCHAR,
                        ah_checksum BYTEA,
                        ax_content VARCHAR,
                        aa_embedding VECTOR(768),
                        ct_valid_from_dt TIMESTAMP,
                        ct_ingest_dt TIMESTAMP,
                        ax_src_system_datastore VARCHAR
                    ) ON COMMIT DROP
                """)

                vals = [
                    (
                        r["kh_knowledge"],
                        r["ax_sub_sequence"],
                        r["ah_checksum"],
                        r.get("ax_content"),
                        r.get("aa_embedding"),
                        r["ct_valid_from_dt"],
                        r["ct_ingest_dt"],
                        r.get("ax_src_system_datastore"),
                    )
                    for r in rows
                ]
                extras.execute_values(c, "INSERT INTO stage_k VALUES %s", vals)

                return self._execute_scd2_logic(
                    c,
                    self.config.knowledge_table,
                    "stage_k",
                    "(kh_knowledge, ct_valid_from_dt, ai_current_flag, ax_sub_sequence, ct_ingest_dt, ax_src_system_datastore, ah_checksum, ax_content, aa_embedding)",
                    "s.kh_knowledge, s.ct_valid_from_dt, 1, s.ax_sub_sequence, s.ct_ingest_dt, s.ax_src_system_datastore, s.ah_checksum, s.ax_content, s.aa_embedding",
                    rows,
                )

    def _ingest_metadata(self, rows: list[dict]) -> dict:
        """Ingesta metadata con SCD2."""
        if not rows:
            return {"updated": 0, "inserted": 0}

        with self.conn:
            with self.conn.cursor() as c:
                c.execute("""
                    CREATE TEMP TABLE stage_m (
                        kh_knowledge BYTEA,
                        ax_sub_sequence VARCHAR,
                        ah_checksum BYTEA,
                        ai_page_number INT,
                        ax_chunk_type JSONB,
                        ax_content VARCHAR,
                        ai_char_count INT,
                        ai_token_count INT,
                        ax_topic VARCHAR,
                        ax_source VARCHAR,
                        ct_valid_from_dt TIMESTAMP,
                        ct_ingest_dt TIMESTAMP,
                        ax_src_system_datastore VARCHAR
                    ) ON COMMIT DROP
                """)

                vals = [
                    (
                        r["kh_knowledge"],
                        r["ax_sub_sequence"],
                        r["ah_checksum"],
                        r.get("ai_page_number"),
                        r.get("ax_chunk_type"),
                        r.get("ax_content"),
                        r.get("ai_char_count"),
                        r.get("ai_token_count"),
                        r.get("ax_topic"),
                        r.get("ax_source"),
                        r["ct_valid_from_dt"],
                        r["ct_ingest_dt"],
                        r.get("ax_src_system_datastore"),
                    )
                    for r in rows
                ]
                extras.execute_values(c, "INSERT INTO stage_m VALUES %s", vals)

                return self._execute_scd2_logic(
                    c,
                    self.config.metadata_table,
                    "stage_m",
                    "(kh_knowledge, ct_valid_from_dt, ai_current_flag, ax_sub_sequence, ct_ingest_dt, ax_src_system_datastore, ah_checksum, ai_page_number, ax_chunk_type, ax_content, ai_char_count, ai_token_count, ax_topic, ax_source)",
                    "s.kh_knowledge, s.ct_valid_from_dt, 1, s.ax_sub_sequence, s.ct_ingest_dt, s.ax_src_system_datastore, s.ah_checksum, s.ai_page_number, s.ax_chunk_type, s.ax_content, s.ai_char_count, s.ai_token_count, s.ax_topic, s.ax_source",
                    rows,
                )

    def _execute_scd2_logic(
        self,
        cursor,
        target_table: str,
        stage_table: str,
        target_cols: str,
        select_cols: str,
        rows: list[dict],
    ) -> dict:
        """Lógica SCD2 con cleanup de chunks huérfanos."""
        # 1. Marcar como históricos los chunks con checksum diferente
        cursor.execute(f"""
            UPDATE {target_table} t
            SET ai_current_flag = 0
            FROM {stage_table} s
            WHERE t.kh_knowledge = s.kh_knowledge
              AND t.ax_sub_sequence = s.ax_sub_sequence
              AND t.ai_current_flag = 1
              AND t.ah_checksum != s.ah_checksum
        """)
        updated = cursor.rowcount

        # 2. Insertar chunks nuevos o modificados
        # ON CONFLICT DO NOTHING maneja duplicados por mismo timestamp o IDs duplicados en origen
        cursor.execute(f"""
            INSERT INTO {target_table} {target_cols}
            SELECT {select_cols}
            FROM {stage_table} s
            WHERE NOT EXISTS (
                SELECT 1 FROM {target_table} t
                WHERE t.kh_knowledge = s.kh_knowledge
                  AND t.ax_sub_sequence = s.ax_sub_sequence
                  AND t.ah_checksum = s.ah_checksum
                  AND t.ai_current_flag = 1
            )
            ON CONFLICT DO NOTHING
        """)
        inserted = cursor.rowcount

        # 3. Soft Delete por Ausencia: desactivar chunks huérfanos
        # (chunks que existían antes pero ya no están en la nueva versión)
        orphans_deactivated = self._deactivate_orphan_chunks(cursor, target_table, rows)

        return {
            "updated": updated,
            "inserted": inserted,
            "orphans_deactivated": orphans_deactivated,
        }

    def _deactivate_orphan_chunks(
        self, cursor, target_table: str, rows: list[dict]
    ) -> int:
        """
        Desactiva chunks huérfanos (Soft Delete por Ausencia).

        Para cada documento procesado, marca como inactivos los chunks que
        existían en la versión anterior pero ya no están en la nueva versión.

        Args:
            cursor: Cursor de base de datos
            target_table: Tabla objetivo (sat_knowledge o sat_knowledge_metadata)
            rows: Lista de chunks procesados en el batch actual

        Returns:
            Cantidad de chunks huérfanos desactivados
        """
        if not rows:
            return 0

        # Agrupar ax_sub_sequence por kh_knowledge (documento)
        docs_subseqs: dict[bytes, list[str]] = {}
        for r in rows:
            kh = r["kh_knowledge"]
            subseq = r["ax_sub_sequence"]
            docs_subseqs.setdefault(kh, []).append(subseq)

        total_deactivated = 0

        for kh_knowledge, processed_subseqs in docs_subseqs.items():
            if not processed_subseqs:
                continue

            # Desactivar chunks que pertenecen al documento pero NO están
            # en la lista de chunks procesados (son huérfanos)
            cursor.execute(
                f"""
                UPDATE {target_table}
                SET ai_current_flag = 0
                WHERE kh_knowledge = %s
                  AND ai_current_flag = 1
                  AND ax_sub_sequence NOT IN %s
                """,
                (kh_knowledge, tuple(processed_subseqs)),
            )
            deactivated = cursor.rowcount
            total_deactivated += deactivated

            if deactivated > 0:
                logger.info(
                    f"   🧹 {deactivated} chunks huérfanos desactivados "
                    f"en {target_table}"
                )

        return total_deactivated


# Helper functions
def _sha1_bytes(s: str) -> bytes:
    return hashlib.sha1(s.encode("utf-8")).digest()


def _to_pgvector(vec: list[float] | None) -> str | None:
    if vec is None:
        return None
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _build_subseq_semantic(chunk_id: str, kh_knowledge: bytes) -> str:
    doc_prefix = kh_knowledge.hex()[:8]
    chunk_num = chunk_id.replace("semantic_chunk_", "").replace("chunk_", "")
    try:
        return f"{doc_prefix}-semantic-chunk-{int(chunk_num):03d}"
    except ValueError:
        return f"{doc_prefix}-{chunk_id}"


def _checksum_semantic(ch: dict) -> bytes:
    """Calculate checksum including both content and embedding_content."""
    return _sha1_bytes(
        json.dumps(
            {
                "id": ch.get("id"),
                "content": ch.get("content", ""),
                "embedding_content": ch.get("embedding_content", ""),
                "section_path": ch.get("section_path", ""),
                "page_start": ch.get("page_start"),
                "page_end": ch.get("page_end"),
            },
            sort_keys=True,
        )
    )
