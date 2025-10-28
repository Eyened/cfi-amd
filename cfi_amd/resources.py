from pathlib import Path
import urllib.request
import tempfile
import shutil
import zipfile
import os
import sys


SENTINELS = {".git", "pyproject.toml"}


def find_repo_root(start: Path) -> Path | None:
    """Walk up from start to find a directory that contains any sentinel.

    Returns the repo root Path or None if not found.
    """
    for parent in [start, *start.parents]:
        for sentinel in SENTINELS:
            if (parent / sentinel).exists():
                return parent
    return None


def default_models_dir() -> Path:
    """Return the default models directory in a centralized user cache.

    Uses platform-specific user cache directory if available; avoids writing into the working directory.
    """
    try:
        from platformdirs import user_cache_dir  # type: ignore
        cache_base = Path(user_cache_dir("cfi-amd", "eyened"))
    except Exception:
        if os.name == "nt":
            base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
            cache_base = Path(base) / "cfi-amd"
        elif sys.platform == "darwin":  # type: ignore
            cache_base = Path.home() / "Library" / "Caches" / "cfi-amd"
        else:
            cache_base = Path.home() / ".cache" / "cfi-amd"
    return cache_base / "models"


def get_models_base_dir(models_dir: str | Path | None = None) -> Path:
    """Resolve the models base directory.

    - If models_dir is provided, use it.
    - Else, try to infer from a git/pyproject repo checkout.
    - If neither works, raise a clear error instructing the user.
    """
    base = Path(models_dir) if models_dir is not None else default_models_dir()
    return base


# Public GitHub release assets used to set up the models folder automatically
ASSETS = [
    {
        "url": "https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/discedge_july24.pt",
        "target": "discedge_july24.pt",
        "is_zip": False,
    },
    {
        "url": "https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/fovea_july24.pt",
        "target": "fovea_july24.pt",
        "is_zip": False,
    },
    {
        "url": "https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/drusen.zip",
        "target": "drusen",
        "is_zip": True,
    },
    {
        "url": "https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/RPD.zip",
        "target": "RPD",
        "is_zip": True,
    },
    {
        "url": "https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/pigment.zip",
        "target": "pigment",
        "is_zip": True,
    },
]


def _stream_download(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # basic console notice
    try:
        print(f"downloading {dest_path.name} from {url}")
    except Exception:
        pass
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore
    with tempfile.NamedTemporaryFile(dir=dest_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(url) as response:  # nosec B310 (trusted URL maintained in code)
            total = int(response.headers.get("Content-Length", 0))
            if tqdm and total > 0:
                with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, desc=dest_path.name) as pbar:  # type: ignore
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
                        pbar.update(len(chunk))  # type: ignore
            else:
                shutil.copyfileobj(response, tmp)
    tmp_path.replace(dest_path)


def _unzip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        # If zip has top-level "models/" directory, strip it to avoid models/models nesting
        def strip_models_prefix(name: str) -> str:
            parts = name.split("/", 1)
            if len(parts) == 2 and parts[0] == "models":
                return parts[1]
            return name
        if names and all(n.startswith("models/") for n in names):
            for member in names:
                target_name = strip_models_prefix(member)
                if not target_name:
                    continue
                dest_path = target_dir / target_name
                if member.endswith("/"):
                    dest_path.mkdir(parents=True, exist_ok=True)
                    continue
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            zf.extractall(target_dir)


def ensure_models_downloaded(models_base_dir: Path) -> None:
    """Ensure required model assets exist under models_base_dir.

    Downloads missing files and unzips archives as needed.
    Safe to call multiple times.
    """
    models_base_dir.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        url = asset["url"]
        target = asset["target"]
        is_zip = asset["is_zip"]

        if is_zip:
            target_dir = models_base_dir / target
            if target_dir.exists():
                continue
            with tempfile.TemporaryDirectory(dir=models_base_dir) as td:
                zip_file = Path(td) / (target + ".zip")
                _stream_download(url, zip_file)
                _unzip(zip_file, models_base_dir)
        else:
            dest_file = models_base_dir / target
            if dest_file.exists():
                continue
            _stream_download(url, dest_file)


