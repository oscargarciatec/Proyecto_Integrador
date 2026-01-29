"""
Cloud Run Job: Procesamiento de PDFs

Uso:
  python -m process_documents.jobs.job_pdf

Variables de entorno:
  - DRIVE_FOLDER_ID: Carpeta de PDFs en Drive
  - DRIVE_CHUNKS_SUBFOLDER: Subcarpeta para chunks (default: "chunks")
  - ALLOYDB_HOST, ALLOYDB_DATABASE, ALLOYDB_USER, ALLOYDB_PASSWORD
"""

import os
import sys

from loguru import logger
from rich.console import Console

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level=os.getenv("LOG_LEVEL", "INFO"),
)

console = Console()


def main():
    from ..processors import ProcessorRegistry
    from ..processors import pdf_processor  # noqa: F401  # Trigger registration

    console.print("\n" + "=" * 60, style="bold blue")
    console.print("🚀 CLOUD RUN JOB: PDF PROCESSOR", style="bold white")
    console.print(
        "   Flujo: Drive PDFs → Docling → Drive chunks/ → AlloyDB", style="dim"
    )
    console.print("=" * 60, style="bold blue")

    dry_run = os.getenv("DRY_RUN", "0").lower() in ("1", "true", "yes")

    processor = ProcessorRegistry.create("pdf", dry_run=dry_run)
    result = processor.run()

    if "error" in result:
        console.print(f"\n❌ Error: {result['error']}", style="red")
        sys.exit(1)

    # Summary
    console.print("\n" + "=" * 60, style="bold green")
    console.print(
        f"🎉 Completado en {result['elapsed_seconds']:.1f}s", style="bold white"
    )
    console.print("=" * 60, style="bold green")


if __name__ == "__main__":
    main()
