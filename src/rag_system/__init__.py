"""Scientific Literature RAG for Document QA.

Import-order note (macOS / Apple silicon) -- please do not "clean this up"
-------------------------------------------------------------------------
``faiss-cpu`` and ``torch`` each bundle their own copy of the LLVM OpenMP
runtime (``libomp.dylib``).  Two copies in one process is undefined behaviour,
and in this stack it reliably produces a bare segmentation fault with no Python
traceback.  Which call crashes depends on the order the two libraries are
loaded:

* faiss loaded first  -> ``AutoModelForSeq2SeqLM.from_pretrained`` (Flan-T5)
  segfaults while loading weights.
* torch loaded first but faiss imported lazily *later* (e.g. the first time a
  FAISS index is read, after ``torch.manual_seed`` has already run) -> the FAISS
  call segfaults.

Both were observed in this project.  The configuration that survives every path
is: **import torch first, then faiss, both eagerly at package import.**  Doing
it here means every entry point inherits the same order regardless of which
submodule it happens to touch first.

``OMP_NUM_THREADS=1`` also avoids the crash, but it serialises both FAISS search
and torch CPU kernels, so it is not used as the primary fix.
"""

from __future__ import annotations

import os as _os

# Fallback guard for environments where the import order alone is insufficient
# (for example a wheel that links OpenMP statically). Harmless otherwise.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Single OpenMP thread: see the module docstring. This is the only setting found
# that survives *every* code path in this stack.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
# Tokenizers' Rust parallelism warns loudly and deadlocks under fork; the
# dataloaders here are single-process, so it buys nothing.
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:  # pragma: no cover - environment dependent
    import torch as _torch  # noqa: F401  MUST precede faiss (see module docstring)
except ImportError:
    _torch = None

try:  # pragma: no cover - environment dependent
    import faiss as _faiss  # noqa: F401
except ImportError:  # faiss is optional for pure-data unit tests
    _faiss = None

__version__ = "0.1.0"
