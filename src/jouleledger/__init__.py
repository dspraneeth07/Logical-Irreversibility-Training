"""JouleLedger: CPU-only, provenance-tagged energy accounting for learning workloads.

Importing this package hides every GPU from the process, unless JOULELEDGER_DEVICE=cuda was set first.
Only the Colab GPU notebook sets it, so the default on any machine is CPU only.
"""
import os

if os.environ.get("JOULELEDGER_DEVICE", "cpu") != "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

__version__ = "0.1.1"
