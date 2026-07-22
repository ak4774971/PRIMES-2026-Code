from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from kaggle.api.kaggle_api_extended import KaggleApi

OUT_DIR = Path(".")  # local folder (current directory)


SOURCES = [
    "https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish",
    "https://fh295.github.io/SimLex-999.zip",
    "https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd",
    "http://download.tensorflow.org/data/questions-words.txt",
]


def _kaggle_slug(url: str) -> str:
    """Extract owner/dataset from a Kaggle dataset URL."""
    match = re.search(r"kaggle\.com/datasets/([^/]+)/([^/?#]+)", url)
    if not match:
        raise ValueError(f"Not a Kaggle dataset URL: {url}")
    return f"{match.group(1)}/{match.group(2)}"


def _download_http(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name or "download.bin"
    dest = dest_dir / filename
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)
    return dest


def _unzip_if_needed(path: Path, dest_dir: Path) -> None:
    if path.suffix.lower() != ".zip":
        return
    print(f"Unzipping {path} -> {dest_dir}")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest_dir)
    path.unlink()  # remove zip after extract
    print(f"Removed {path.name}")


def download_and_unzip(url: str, dest_dir: Path = OUT_DIR) -> None:
    """Download a URL into dest_dir; unzip if it is a zip/Kaggle dataset archive."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if "kaggle.com/datasets/" in url:
        slug = _kaggle_slug(url)
        print(f"Downloading Kaggle dataset {slug} -> {dest_dir}")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            slug,
            path=str(dest_dir),
            unzip=True,
            quiet=False,
        )
        # Clean leftover zip(s) if any remain
        for zip_path in dest_dir.glob("*.zip"):
            # Only remove zips that belong to this dataset download naming
            if slug.split("/")[-1] in zip_path.stem or zip_path.stem:
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.testzip()
                    zip_path.unlink()
                    print(f"Removed leftover {zip_path.name}")
                except zipfile.BadZipFile:
                    pass
        return

    # Direct HTTP(S) file
    path = _download_http(url, dest_dir)
    _unzip_if_needed(path, dest_dir)


def main() -> None:
    for url in SOURCES:
        print("\n" + "=" * 60)
        download_and_unzip(url, OUT_DIR)

    print("\nDone. Files in:", OUT_DIR.resolve())
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT_DIR)}")


if __name__ == "__main__":
    main()