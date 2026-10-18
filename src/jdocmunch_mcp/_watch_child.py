"""Run ONE watcher reindex batch in a throwaway interpreter.

Why a process and not a thread: the incremental refresh rewrites the whole
monolith however few files changed (``DocStore.incremental_save`` is a full
``json.load`` -> posting ``RowSet`` re-tokenization over every section ->
``json.dump`` rewrite), so each event's peak is the largest index's full
rebuild, and the allocator floor it leaves behind (glibc arenas for
``asyncio.to_thread``'s pool plus partially-used pymalloc arenas) never
fully returns: measured on a 90k-section corpus, a watch daemon 9 h old sat
at 1.34 GB RSS with a live set near zero. In a child process the peak is
paid and every page goes back to the OS when the child exits, so the watch
parent stays at its baseline whatever the corpus size or event rate.

Invoked only by ``watch._handle_changes``; the argv contract is internal
and deliberately NOT a public CLI surface:

    python -m jdocmunch_mcp._watch_child <root> <name> <storage_path> <ai:0|1>

with the changed paths on stdin, one per line (the same convention as the
CLI's ``index-local --paths-from -``). Exit 0 on success; an exception
prints its traceback on stderr and exits nonzero, which the parent surfaces
through its existing "reindex failed" warning exactly as an in-process
exception would.
"""
from __future__ import annotations

import json
import sys


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 4:
        print(
            "usage: python -m jdocmunch_mcp._watch_child "
            "<root> <name> <storage_path> <ai:0|1>",
            file=sys.stderr,
        )
        return 2
    root, name, storage_path, ai = argv
    paths = [line for line in sys.stdin.read().splitlines() if line.strip()]
    if not paths:
        return 0
    from .tools.index_local import index_local

    result = index_local(
        path=root,
        name=name,
        paths=paths,
        storage_path=storage_path,
        use_ai_summaries=ai == "1",
        incremental=True,
    )
    # index_local reports failures as an error result rather than raising
    # (e.g. {"success": false, "error": "Invalid name: ..."}). Without this
    # check the child would exit 0 and the parent would log "re-indexed N
    # file(s)" over a batch that did nothing.
    if isinstance(result, dict) and result.get("success") is False:
        print(json.dumps(result), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
