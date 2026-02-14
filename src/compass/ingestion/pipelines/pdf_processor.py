"""
PDF document processor.

Handles PDF processing with Docling and chunk ingestion to AlloyDB.
"""

import io
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from .base import BaseProcessor, ProcessorResult
from .registry import ProcessorRegistry


def _get_or_create_chunks_folder(
    drive_service, parent_folder_id: str, folder_name: str = "chunks"
) -> str:
    """Obtiene o crea la subcarpeta de chunks dentro de la carpeta de PDFs."""
    from ..sources.drive_service import drive_list_all

    query = f"'{parent_folder_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"

    folders = drive_list_all(
        drive_service,
        q=query,
        fields="files(id, name)",
    )

    if folders:
        folder_id = folders[0].get("id")
        logger.info(f"📁 Subcarpeta '{folder_name}' encontrada: {folder_id}")
        return folder_id

    # Crear si no existe
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }

    folder = (
        drive_service.files()
        .create(
            body=file_metadata,
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )

    folder_id = folder.get("id")
    logger.info(f"📁 Subcarpeta '{folder_name}' creada: {folder_id}")
    return folder_id


@ProcessorRegistry.register("pdf")
class PDFProcessor(BaseProcessor):
    """
    Procesador de documentos PDF.

    Pipeline:
      1. Detectar PDFs en carpeta de Drive
      2. Procesar con Docling → chunks semánticos
      3. Guardar JSONs en subcarpeta 'chunks/'
      4. Generar embeddings e ingestar a AlloyDB
    """

    @property
    def name(self) -> str:
        return "pdf"

    def __init__(self, dry_run: bool = False):
        super().__init__(dry_run)
        self.folder_id = os.getenv("DRIVE_FOLDER_ID")
        self.chunks_subfolder = os.getenv("DRIVE_CHUNKS_SUBFOLDER", "chunks")
        self.archive_folder_id = os.getenv("DRIVE_ARCHIVE_FOLDER_ID")
        self.force_reprocess = os.getenv("FORCE_REPROCESS", "0").lower() in (
            "1",
            "true",
            "yes",
        )

    def validate(self) -> bool:
        if not self.folder_id:
            logger.error("DRIVE_FOLDER_ID no configurado")
            return False
        return True

    def process(self) -> ProcessorResult:
        """Procesa PDFs desde Drive y genera chunks."""
        from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

        from ..sources.drive_service import get_drive_service, drive_list_all
        from ..core.document_processor import PolicyDocumentProcessor
        from ..core.settings import ChunkingConfig, ChunkMetadata

        result = ProcessorResult()

        logger.info(f"📂 Carpeta PDFs: {self.folder_id}")
        logger.info(f"📁 Subcarpeta chunks: {self.chunks_subfolder}/")

        try:
            drive_service = get_drive_service()
            chunks_folder_id = _get_or_create_chunks_folder(
                drive_service, self.folder_id, self.chunks_subfolder
            )

            config = ChunkingConfig()
            processor = PolicyDocumentProcessor(config)

            # Listar PDFs
            query = f"'{self.folder_id}' in parents and mimeType='application/pdf' and trashed=false"
            pdf_items = drive_list_all(
                drive_service,
                q=query,
                fields="files(id, name, md5Checksum, modifiedTime, appProperties)",
            )

            logger.info(f"📄 PDFs encontrados: {len(pdf_items)}")

            # Listar JSONs existentes
            chunks_query = f"'{chunks_folder_id}' in parents and trashed=false"
            existing_chunks = drive_list_all(
                drive_service,
                q=chunks_query,
                fields="files(id, name, appProperties)",
            )
            existing_chunks_by_name = {c.get("name"): c for c in existing_chunks}

            for item in pdf_items:
                file_id = item.get("id")
                file_name = item.get("name")
                file_md5 = item.get("md5Checksum")
                app_props = item.get("appProperties") or {}

                json_filename = f"{Path(file_name).stem}_chunks.json"

                # Verificar si ya fue procesado
                if not self.force_reprocess:
                    processed_md5 = app_props.get("processed_md5")
                    if processed_md5 and processed_md5 == file_md5:
                        logger.debug(f"⏭️ Skip: {file_name} ya procesado")
                        result.skipped += 1
                        continue

                logger.info(f"🚀 Procesando: {file_name}")
                tmp_path = None

                try:
                    # Descargar PDF a temporal
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        request = drive_service.files().get_media(
                            fileId=file_id, supportsAllDrives=True
                        )
                        downloader = MediaIoBaseDownload(tmp, request)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                        tmp_path = tmp.name

                    # Procesar con Docling
                    start_time = datetime.now()
                    chunks = processor.process_pdf(tmp_path, source_name=file_name)
                    elapsed = (datetime.now() - start_time).total_seconds()

                    if not chunks:
                        logger.warning(f"⚠️ No se generaron chunks para {file_name}")
                        continue

                    # Preparar JSON
                    token_counts = [c["token_count"] for c in chunks]
                    total_tokens = sum(token_counts)

                    metadata = ChunkMetadata(
                        source_pdf=file_name,
                        total_chunks=len(chunks),
                        total_tokens=total_tokens,
                        avg_tokens=total_tokens / len(chunks) if chunks else 0,
                        min_chunk_tokens=min(token_counts) if chunks else 0,
                        max_chunk_tokens=max(token_counts) if chunks else 0,
                        pages_skipped=0,
                        config=asdict(config),
                    )

                    result_json = {"metadata": asdict(metadata), "chunks": chunks}
                    json_bytes = json.dumps(
                        result_json, indent=2, ensure_ascii=False
                    ).encode("utf-8")

                    # Subir JSON a subcarpeta de Drive
                    existing_chunk = existing_chunks_by_name.get(json_filename)
                    existing_chunk_id = (
                        existing_chunk.get("id") if existing_chunk else None
                    )

                    media = MediaIoBaseUpload(
                        io.BytesIO(json_bytes),
                        mimetype="application/json",
                        resumable=False,
                    )

                    chunk_app_props = {
                        "source_pdf_id": file_id,
                        "source_pdf_md5": str(file_md5 or ""),
                        "source_pdf_name": file_name,
                        # Clear ingestion flag to force re-ingestion
                        # (Drive API merges appProperties, doesn't replace)
                        "ingested_md5": "",
                    }

                    if existing_chunk_id:
                        drive_service.files().update(
                            fileId=existing_chunk_id,
                            body={
                                "name": json_filename,
                                "appProperties": chunk_app_props,
                            },
                            media_body=media,
                            fields="id",
                            supportsAllDrives=True,
                        ).execute()
                        logger.info(f"   📝 Actualizado: {json_filename}")
                    else:
                        file_body = {
                            "name": json_filename,
                            "parents": [chunks_folder_id],
                            "appProperties": chunk_app_props,
                        }
                        drive_service.files().create(
                            body=file_body,
                            media_body=media,
                            fields="id",
                            supportsAllDrives=True,
                        ).execute()
                        logger.info(f"   ✅ Creado: {json_filename}")

                    logger.info(
                        f"   📊 {len(chunks)} chunks | {total_tokens} tokens | {elapsed:.1f}s"
                    )

                    # Marcar PDF como procesado
                    new_props = dict(app_props)
                    new_props.update(
                        {
                            "processed_md5": str(file_md5 or ""),
                            "processed_at": datetime.utcnow().isoformat() + "Z",
                        }
                    )

                    drive_service.files().update(
                        fileId=file_id,
                        body={"appProperties": new_props},
                        fields="id",
                        supportsAllDrives=True,
                    ).execute()

                    # Mover PDF a carpeta ARCHIVE si está configurada
                    if self.archive_folder_id:
                        try:
                            from ..sources.drive_service import drive_move_file

                            drive_move_file(
                                drive_service,
                                file_id=file_id,
                                new_parent_id=self.archive_folder_id,
                                old_parent_id=self.folder_id,
                            )
                            logger.info(f"   📦 Archivado: {file_name} → ARCHIVE")
                        except Exception as e:
                            logger.warning(f"   ⚠️ No se pudo archivar {file_name}: {e}")

                    result.processed += 1

                except Exception as e:
                    error_msg = f"{file_name}: {str(e)}"
                    logger.error(f"❌ Error: {error_msg}")
                    result.errors.append(error_msg)

                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            logger.info(
                f"✅ Procesamiento: {result.processed} nuevos, {result.skipped} existentes"
            )

        except Exception as e:
            logger.exception(f"Error general: {e}")
            result.errors.append(str(e))

        return result

    def ingest(self) -> ProcessorResult:
        """Ingesta chunks a AlloyDB con embeddings."""
        from ..core.ingestion_drive import DriveChunkIngester, IngestionConfig

        config = IngestionConfig(
            drive_pdf_folder_id=self.folder_id,
            drive_chunks_subfolder=self.chunks_subfolder,
            project_id=os.getenv("GCS_PROJECT_ID", os.getenv("GCP_PROJECT_ID", "")),
            vertex_location=os.getenv("VERTEX_LOCATION", "us-east1"),
            alloydb_host=os.getenv("ALLOYDB_HOST"),
            alloydb_port=int(os.getenv("ALLOYDB_PORT", "5432")),
            alloydb_database=os.getenv("ALLOYDB_DATABASE"),
            alloydb_user=os.getenv("ALLOYDB_USER"),
            alloydb_password=os.getenv("ALLOYDB_PASSWORD"),
            alloydb_sslmode=os.getenv("ALLOYDB_SSLMODE", "require"),
        )

        logger.info(
            f"📂 Leyendo chunks de Drive: {self.folder_id}/{self.chunks_subfolder}/"
        )
        logger.info(f"💾 Destino AlloyDB: {config.alloydb_host}")

        try:
            ingester = DriveChunkIngester(config)
            raw_result = ingester.run()

            logger.info(f"✅ Ingesta completada: {raw_result}")

            return ProcessorResult(
                inserted=raw_result.get("chunks_inserted", 0),
                updated=raw_result.get("chunks_updated", 0),
                errors=[raw_result.get("error")] if raw_result.get("error") else [],
            )

        except Exception as e:
            logger.exception(f"Error en ingesta: {e}")
            return ProcessorResult(errors=[str(e)])
