#!/usr/bin/env python
"""Minimal terminal demo: ask the local RAG stack a question and show its sources.

    python app/demo.py --question "What is the main cause of HIV-1 infection in children?"
    python app/demo.py --interactive

Runs the pipeline in-process (no HTTP server needed).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_system  # noqa: F401,E402

from rag_system.api.service import RAGService  # noqa: E402

SAMPLE_QUESTIONS = [
    "What is the main cause of HIV-1 infection in children?",
    "What is the most common species of Human Coronavirus among adults?",
    "How is the basic reproduction number estimated?",
]


def show(service: RAGService, question: str, top_k: int) -> None:
    response = service.pipeline.query(question, top_k=top_k)
    print("\n" + "=" * 88)
    print(f"Q: {question}")
    print("-" * 88)
    print(f"A: {response.answer or '(no answer found)'}")
    print(f"\nlatency: " + "  ".join(f"{k}={v:.0f}ms" for k, v in response.latency_ms.items()))
    print(f"\nsources ({len(response.contexts)} chunks):")
    for c in response.contexts:
        snippet = " ".join(c.text.split())[:220]
        title = c.metadata.get("title", "")[:70]
        print(f"  [{c.rank}] {c.chunk_id}  score={c.score:.3f}")
        if title:
            print(f"      doc: {title}")
        print(textwrap.fill(snippet, width=84, initial_indent="      ", subsequent_indent="      "))
    print("=" * 88)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--question", "-q", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--adapter", default=None, help="path to a LoRA adapter directory")
    p.add_argument("--interactive", "-i", action="store_true")
    args = p.parse_args()

    service = RAGService(adapter_path=args.adapter)

    if args.interactive:
        print("Type a question (blank line or Ctrl-D to exit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
            show(service, q, args.top_k)
        return

    for q in ([args.question] if args.question else SAMPLE_QUESTIONS):
        show(service, q, args.top_k)


if __name__ == "__main__":
    main()
