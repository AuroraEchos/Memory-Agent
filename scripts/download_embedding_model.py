"""Download the configured embedding model into the shared Docker volume."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_SOURCE = "BAAI/bge-m3"
DEFAULT_MODEL_PATH = "/app/models/current"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
MODEL_INFO_FILENAME = ".model-info.json"


def load_model_source(output_dir: Path) -> str | None:
    """Read downloader metadata when a previous download recorded it."""

    info_path = output_dir / MODEL_INFO_FILENAME
    if not info_path.exists():
        return None

    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    source = payload.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()

    return None


def write_model_info(output_dir: Path, source: str) -> None:
    """Persist source metadata alongside the downloaded model."""

    info_path = output_dir / MODEL_INFO_FILENAME
    info_path.write_text(
        json.dumps({"source": source}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def is_model_ready(output_dir: Path, source: str) -> bool:
    """Return whether the target directory already contains the requested model."""

    if not (output_dir / "modules.json").exists():
        return False

    existing_source = load_model_source(output_dir)
    if existing_source is None:
        print(
            "Embedding model already exists and has no downloader metadata; "
            "reusing it."
        )
        return True

    return existing_source == source


def main() -> int:
    """Download the configured embedding model when it is missing."""

    source = os.getenv("EMBEDDING_MODEL_SOURCE", DEFAULT_MODEL_SOURCE).strip()
    output_dir = Path(
        os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_PATH).strip()
    ).resolve()
    hf_endpoint = os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).strip()

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if is_model_ready(output_dir, source):
        print(f"Embedding model is ready at {output_dir} from {source}.")
        return 0

    if output_dir.exists():
        print(f"Replacing existing embedding model directory at {output_dir}.")

    with tempfile.TemporaryDirectory(
        prefix=f"{output_dir.name}-",
        dir=str(output_dir.parent),
    ) as temp_dir:
        temp_path = Path(temp_dir)
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            print(f"Using Hugging Face mirror: {hf_endpoint}")
        print(f"Downloading embedding model {source} to {output_dir}...")
        model = SentenceTransformer(source, device="cpu")
        model.save(str(temp_path))
        write_model_info(temp_path, source)

        if output_dir.exists():
            shutil.rmtree(output_dir)

        shutil.move(str(temp_path), str(output_dir))

    print(f"Embedding model downloaded to {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
