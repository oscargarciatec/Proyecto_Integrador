"""
Servicio de integración con Google Drive para el procesamiento de documentos.
Maneja autenticación, listado, descarga y subida de archivos.
"""

import os
import io
import json
import tempfile
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Optional

try:
    from google.oauth2 import service_account
except ImportError:
    service_account = None

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False

from ..core.settings import (
    SERVICE_ACCOUNT_FILE,
    DRIVE_FOLDER_ID,
    DRIVE_SCOPES,
    ChunkingConfig,
    ChunkMetadata,
)
# NOTE: PolicyDocumentProcessor is imported lazily inside process_drive_folder
# to avoid loading docling for Excel jobs


def get_drive_service():
    """
    Crea y retorna un servicio autenticado de Google Drive.

    En Cloud Run usa ADC (Application Default Credentials).
    En desarrollo local usa el archivo de Service Account.
    """
    if not DRIVE_AVAILABLE:
        raise ImportError(
            "google-api-python-client no está instalado. Ejecuta: pip install google-api-python-client"
        )

    # Verificar si existe el archivo de Service Account
    sa_file = os.getenv("SERVICE_ACCOUNT_FILE", SERVICE_ACCOUNT_FILE)

    if sa_file and os.path.exists(sa_file):
        # Usar Service Account file (desarrollo local)
        if not service_account:
            from google.oauth2 import service_account as sa

            creds_cls = sa.Credentials
        else:
            creds_cls = service_account.Credentials

        creds = creds_cls.from_service_account_file(sa_file, scopes=DRIVE_SCOPES)
    else:
        # Usar ADC (Cloud Run, GCE, etc.)
        from google.auth import default

        creds, _ = default(scopes=DRIVE_SCOPES)

    return build("drive", "v3", credentials=creds)


def drive_list_all(
    service, *, q: str, fields: str, page_size: int = 1000
) -> List[Dict]:
    """Lista archivos con paginación y soporte para Shared Drives."""
    files = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=q,
                fields=f"nextPageToken, {fields}",
                pageSize=page_size,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                corpora="allDrives",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def drive_upsert_json(
    service,
    *,
    folder_id: str,
    filename: str,
    content_bytes: bytes,
    app_properties: Dict[str, str],
    existing_file_id: Optional[str],
) -> Dict:
    """Crea o actualiza un archivo JSON en Drive."""
    media = MediaIoBaseUpload(
        io.BytesIO(content_bytes), mimetype="application/json", resumable=False
    )
    if existing_file_id:
        body = {
            "name": filename,
            "appProperties": app_properties,
        }
        return (
            service.files()
            .update(
                fileId=existing_file_id,
                body=body,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
    body = {
        "name": filename,
        "parents": [folder_id],
        "appProperties": app_properties,
    }
    return (
        service.files()
        .create(
            body=body,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )


def process_drive_folder(folder_id: str = None, config: ChunkingConfig = None):
    """Procesa PDFs desde una carpeta de Google Drive y guarda JSONs ahí mismo."""
    # Lazy import to avoid loading docling for Excel jobs
    from ..core.document_processor import PolicyDocumentProcessor

    folder_id = folder_id or DRIVE_FOLDER_ID
    config = config or ChunkingConfig()
    processor = PolicyDocumentProcessor(config)

    try:
        service = get_drive_service()
        print(f"📡 Conectado a Google Drive. Carpeta ID: {folder_id}")

        force_reprocess = os.getenv("FORCE_REPROCESS", "0").strip() in (
            "1",
            "true",
            "True",
            "yes",
            "YES",
        )

        # Listar items
        query = f"'{folder_id}' in parents and trashed=false"
        all_items = drive_list_all(
            service,
            q=query,
            fields="files(id, name, mimeType, shortcutDetails, md5Checksum, modifiedTime, size, appProperties)",
        )

        if not all_items:
            print("⚠️ La carpeta existe/accesible pero Drive devolvió 0 items.")
            print(
                "   - Verifica que la carpeta esté compartida con la Service Account (Editor o al menos Lector)."
            )
            return

        print(f"📄 Items visibles en la carpeta: {len(all_items)}")
        for it in all_items[:20]:
            mt = it.get("mimeType")
            print(f"   - {it.get('name')} | {mt} | id={it.get('id')}")

        pdf_items = []
        chunks_items = []
        for it in all_items:
            mt = it.get("mimeType")
            name = (it.get("name") or "").strip()

            if mt == "application/pdf":
                pdf_items.append(it)
                continue

            # Soportar shortcuts
            if mt == "application/vnd.google-apps.shortcut":
                sd = it.get("shortcutDetails") or {}
                if sd.get("targetMimeType") == "application/pdf" and sd.get("targetId"):
                    try:
                        target = (
                            service.files()
                            .get(
                                fileId=sd.get("targetId"),
                                fields="id, name, mimeType, md5Checksum, modifiedTime, size, appProperties",
                                supportsAllDrives=True,
                            )
                            .execute()
                        )
                        target["name"] = name or target.get("name")
                        pdf_items.append(target)
                    except Exception:
                        pdf_items.append(
                            {
                                "id": sd.get("targetId"),
                                "name": name,
                                "mimeType": "application/pdf",
                            }
                        )
                continue

            if name.lower().endswith("_chunks.json"):
                chunks_items.append(it)

        if not pdf_items:
            print("⚠️ No se encontraron PDFs (ni shortcuts a PDFs) en la carpeta.")
            _diagnose_missing_pdfs(service)
            return

        # Index de chunks existentes
        chunks_by_name: Dict[str, List[Dict]] = {}
        for ch in chunks_items:
            n = (ch.get("name") or "").strip()
            chunks_by_name.setdefault(n, []).append(ch)

        def _pick_latest(files: List[Dict]) -> Optional[Dict]:
            if not files:
                return None
            return sorted(
                files, key=lambda x: x.get("modifiedTime") or "", reverse=True
            )[0]

        # Deduplicar PDFs por nombre
        pdf_by_name: Dict[str, List[Dict]] = {}
        for p in pdf_items:
            n = (p.get("name") or "").strip()
            pdf_by_name.setdefault(n, []).append(p)
        pdf_items = [
            _pick_latest(v) for v in pdf_by_name.values() if _pick_latest(v) is not None
        ]

        print(f"📄 PDFs detectados: {len(pdf_items)}")

        for item in pdf_items:
            _process_single_pdf(
                service,
                item,
                folder_id,
                processor,
                config,
                chunks_by_name,
                force_reprocess,
            )

    except Exception as e:
        print(f"❌ Error general en Drive: {e}")


def _diagnose_missing_pdfs(service):
    """Diagnóstico cuando no se encuentran PDFs."""
    print(
        "   - Si el PDF aparece en tu UI pero no acá, casi seguro es un tema de permisos o Shared Drive."
    )
    print(
        "   - Diagnóstico extra: buscando PDFs accesibles por la Service Account con nombre similar..."
    )

    try:
        guess = "politica"
        query2 = (
            f"mimeType='application/pdf' and trashed=false and name contains '{guess}'"
        )
        results2 = (
            service.files()
            .list(
                q=query2,
                fields="files(id, name, parents, driveId)",
                pageSize=20,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                corpora="allDrives",
            )
            .execute()
        )
        found = results2.get("files", [])
        if not found:
            print(
                "   - La Service Account no ve PDFs con 'politica' en el nombre (probable falta de permisos al PDF)."
            )
        else:
            print(
                f"   - PDFs encontrados globalmente: {len(found)} (primeros {min(20, len(found))})"
            )
            for f in found:
                print(
                    f"     • {f.get('name')} | id={f.get('id')} | parents={f.get('parents')} | driveId={f.get('driveId')}"
                )
            print(
                "   - Si tu PDF no aparece acá: compártelo explícitamente con la Service Account o súbelo de nuevo dentro de la carpeta."
            )
    except Exception as e:
        print(f"   - Diagnóstico falló: {e}")


def _process_single_pdf(
    service,
    item: Dict,
    folder_id: str,
    processor,  # PolicyDocumentProcessor (lazy import)
    config: ChunkingConfig,
    chunks_by_name: Dict[str, List[Dict]],
    force_reprocess: bool,
):
    """Procesa un único PDF de Drive."""
    file_id = item.get("id")
    file_name = item.get("name")
    file_md5 = item.get("md5Checksum")
    file_mtime = item.get("modifiedTime")
    pdf_props = item.get("appProperties") or {}

    print(f"\n{'=' * 60}")
    print(f"🚀 Procesando: {file_name} (ID: {file_id})")

    # --- SKIP/UPSERT LOGIC ---
    stem = Path(file_name).stem
    json_filename = f"{stem}_chunks.json"

    def _pick_latest(files: List[Dict]) -> Optional[Dict]:
        if not files:
            return None
        return sorted(files, key=lambda x: x.get("modifiedTime") or "", reverse=True)[0]

    existing_chunk = _pick_latest(chunks_by_name.get(json_filename, []))
    existing_id = existing_chunk.get("id") if existing_chunk else None
    existing_props = (
        (existing_chunk.get("appProperties") or {}) if existing_chunk else {}
    )
    existing_mtime = existing_chunk.get("modifiedTime") if existing_chunk else None

    already_processed = False
    if existing_chunk and not force_reprocess:
        processed_md5 = pdf_props.get("processed_md5")
        processed_mtime = pdf_props.get("processed_mtime")
        if processed_md5 and file_md5 and processed_md5 == file_md5:
            already_processed = True
        elif processed_mtime and file_mtime and processed_mtime == file_mtime:
            already_processed = True
        elif (
            existing_props.get("source_pdf_id") == file_id
            and file_md5
            and existing_props.get("source_pdf_md5") == file_md5
        ):
            already_processed = True
        elif file_mtime and existing_mtime and existing_mtime >= file_mtime:
            already_processed = True

    if already_processed:
        print(f"   ⏭️  Skip: ya existe '{json_filename}' para este PDF (sin cambios).")
        return

    # Descargar PDF
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        downloader = MediaIoBaseDownload(tmp, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        tmp_path = tmp.name

    try:
        # Procesar
        start_time = datetime.now()
        chunks = processor.process_pdf(tmp_path, source_name=file_name)
        elapsed = (datetime.now() - start_time).total_seconds()

        if not chunks:
            print("   ⚠️ No se generaron chunks")
            return

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

        result = {"metadata": asdict(metadata), "chunks": chunks}

        # Subir JSON a Drive
        json_bytes = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
        app_props = {
            "source_pdf_id": str(file_id or ""),
            "source_pdf_md5": str(file_md5 or ""),
            "source_pdf_name": str(file_name or ""),
        }
        uploaded_file = drive_upsert_json(
            service,
            folder_id=folder_id,
            filename=json_filename,
            content_bytes=json_bytes,
            app_properties=app_props,
            existing_file_id=existing_id,
        )

        # Marcar PDF como procesado
        try:
            new_pdf_props = dict(pdf_props)
            new_pdf_props.update(
                {
                    "processed_md5": str(file_md5 or ""),
                    "processed_mtime": str(file_mtime or ""),
                    "processed_chunks_file_id": str(uploaded_file.get("id") or ""),
                    "processed_chunks_name": str(json_filename),
                    "processed_at": datetime.utcnow().replace(microsecond=0).isoformat()
                    + "Z",
                    "processed_strategy": "docling_section_based_v2",
                }
            )
            service.files().update(
                fileId=file_id,
                body={"appProperties": new_pdf_props},
                fields="id",
                supportsAllDrives=True,
            ).execute()
        except Exception:
            pass

        print(f"   ✅ Subido a Drive: {json_filename} (ID: {uploaded_file.get('id')})")
        print(
            f"      Chunks: {len(chunks)} | Tokens: {total_tokens} | Tiempo: {elapsed:.1f}s"
        )

    except Exception as e:
        print(f"   ❌ Error procesando {file_name}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


__all__ = [
    "DRIVE_AVAILABLE",
    "get_drive_service",
    "drive_list_all",
    "drive_upsert_json",
    "process_drive_folder",
]
