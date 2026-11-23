"""Recompute the preregistration hash: SHA-256 of the frozen block of the preregistration, a separator, and
the configuration it was frozen with (line endings normalised to LF). It must equal the value in metadata.yaml.

    python verify_preregistration.py
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAIRS = [
    ("PREREGISTRATION.md", "config.yaml", "56867144e8a3611588225dd86c0ff79235f349f1df589afe0ffacadd86dca591"),
]


def lf(path):
    return (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")


def digest(prereg, config):
    block = lf(prereg).split("<!-- FROZEN:BEGIN -->")[1].split("<!-- FROZEN:END -->")[0]
    return hashlib.sha256((block + "\n---\n" + lf(config)).encode("utf-8")).hexdigest()


ok = True
for prereg, config, recorded in PAIRS:
    h = digest(prereg, config)
    print(f"{prereg} + {config}: {h} {'matches' if h == recorded else 'DOES NOT MATCH'}")
    ok = ok and h == recorded
sys.exit(0 if ok else 1)
