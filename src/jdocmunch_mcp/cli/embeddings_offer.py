"""`init` step: offer offline embeddings (semantic search) once, default No.

Why this exists: a default install has no embedding provider, so
``use_embeddings="auto"`` resolves to lexical and the user never learns that
hybrid search exists. On the public benchmark below, hybrid put a direct answer
in the top 5 for 47% of questions against 34% for lexical.

⚠⚠ This is the ONE place jdocmunch runs pip or starts a download on the user's
behalf, and it does so only on an explicit yes (interactive) or an explicit
``--with-embeddings`` (scripted). ``--yes`` does NOT opt in: a flag that
pre-approves config edits must not also pre-approve a ~250 MB download. The
server itself never installs or downloads anything unasked. README, "Background
behavior, fully disclosed", carries the matching disclosure.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Callable, Optional

# --- Figures quoted to the user -------------------------------------------
# Source: github.com/jgravelle/jdoc-rerank-bench, results/baseline-gap-2026-09-19.md
# and results/fastembed-setup-2026-09-19.md. tests/test_init_embeddings_offer.py
# RESTATES these rather than importing them; a pin that reads the value it checks
# asserts nothing.
BENCH_URL = "https://github.com/jgravelle/jdoc-rerank-bench"
BENCH_QUESTIONS = 492
BENCH_DOC_SETS = 6
BENCH_TOP5_LEXICAL_PCT = 34
BENCH_TOP5_HYBRID_PCT = 47
FASTEMBED_INSTALL_MB = 160      # venv grew from 9 MB to 169 MB on a cold install
MODEL_DOWNLOAD_MB = 87          # all-MiniLM-L6-v2, ONNX, measured on disk
SECONDS_PER_1000_SECTIONS = 7   # measured 6.1 to 7.5 on five corpora, one CPU box

FASTEMBED_REQUIREMENT = "fastembed>=0.8.0"


def pip_command() -> list[str]:
    """The exact command `init` runs on a yes. Printed before it runs."""
    return [sys.executable, "-m", "pip", "install", FASTEMBED_REQUIREMENT]


def offer_text() -> str:
    """The prompt shown once when no embedding provider resolves."""
    return (
        "Semantic search is off. Searches will match words only.\n"
        f"On our benchmark ({BENCH_QUESTIONS} questions, {BENCH_DOC_SETS} doc sets), turning it on put a direct\n"
        f"answer in the top 5 results for {BENCH_TOP5_HYBRID_PCT}% of questions, up from {BENCH_TOP5_LEXICAL_PCT}%.\n"
        f"  {BENCH_URL}\n"
        f"To turn it on, jdocmunch needs the `fastembed` package (about {FASTEMBED_INSTALL_MB} MB installed)\n"
        f"and one model file ({MODEL_DOWNLOAD_MB} MB, downloaded once from huggingface.co).\n"
        "Everything then runs on this machine. No text from your documents is sent anywhere.\n"
        "Install it now?"
    )


def manual_instructions() -> str:
    """How to turn it on by hand, for each way jdocmunch gets installed."""
    return (
        "  To turn semantic search on later, install the offline provider and re-index:\n"
        '    pip:     pip install "jdocmunch-mcp[fastembed]"\n'
        '    uv tool: uv tool install "jdocmunch-mcp[fastembed]" --force\n'
        "    pipx:    pipx inject jdocmunch-mcp fastembed\n"
        "    then:    jdocmunch-mcp index-local --path <your docs folder>\n"
        "  Details: README, \"Background behavior, fully disclosed\"."
    )


def _resolved_provider() -> Optional[str]:
    try:
        from ..embeddings.provider import get_provider_name
        return get_provider_name()
    except Exception:
        return None


def _install_fastembed(run: Callable = subprocess.run) -> tuple[bool, str]:
    """Run pip in THIS interpreter. Never raises; returns (ok, detail)."""
    cmd = pip_command()
    try:
        probe = run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True,
                    encoding="utf-8", errors="replace")
        if probe.returncode != 0:
            return False, "this Python environment has no pip (uv tool and pipx installs do not)"
        done = run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or "").strip().splitlines()[-1:] or ["pip failed"]
            return False, tail[0][:200]
    except Exception as exc:  # locked-down environments raise here, not in pip
        return False, str(exc)[:200]
    importlib.invalidate_caches()
    return True, ""


def _download_model() -> tuple[bool, str]:
    """Fetch the model into jdocmunch's stable model folder, through the
    provider's own constructor. The server loads from that folder once the
    model is there, so the file lands where it will look. Never raises.

    ⚠ The stable folder, not FastEmbed's ``<tempdir>/fastembed_cache``: OS
    cleanup can empty the temp dir, and that is the damage a repair fixes.
    """
    try:
        from ..embeddings import provider as _p
        _p._FastEmbedProvider(cache_dir=str(_p._stable_fastembed_cache()))
        # A load that failed earlier in this process must not stand.
        _p._PROVIDER_ERRORS.clear()
        return True, ""
    except Exception as exc:
        return False, str(exc)[:200]


def _model_problem() -> Optional[str]:
    try:
        from ..embeddings.provider import fastembed_model_problem
        return fastembed_model_problem()
    except Exception as exc:
        return str(exc)[:200]


def repair_text(problem: str) -> str:
    """The prompt shown when FastEmbed is installed but its model won't load."""
    return (
        "Semantic search is configured (fastembed), but its model is missing or damaged:\n"
        f"  {problem}\n"
        "Until it is repaired, searches match words only.\n"
        f"Repairing downloads one model file ({MODEL_DOWNLOAD_MB} MB, once, from huggingface.co)\n"
        "into jdocmunch's own model folder, outside the temp directory.\n"
        "Repair it now?"
    )


def offer_embeddings(
    *,
    interactive: bool,
    with_embeddings: bool,
    dry_run: bool,
    prompt_yn: Callable[[str, bool], bool],
    out: Callable[[str], None] = print,
    run: Callable = subprocess.run,
) -> bool:
    """Run the offer. Returns True when the following index step should embed.

    Every path that does not end in a working provider leaves the install
    lexical and says how to change that. Nothing here raises.
    """
    provider = _resolved_provider()
    if provider == "fastembed":
        problem = _model_problem()
        if problem:
            return _offer_repair(
                problem, interactive=interactive, with_embeddings=with_embeddings,
                dry_run=dry_run, prompt_yn=prompt_yn, out=out,
            )
    if provider:
        out(f"  Semantic search: on ({provider})")
        return True

    if dry_run:
        out("  Semantic search: off. With --with-embeddings (or a yes at the prompt) init would run:")
        out(f"    {' '.join(pip_command())}   (about {FASTEMBED_INSTALL_MB} MB)")
        out(f"    then download one model file from huggingface.co ({MODEL_DOWNLOAD_MB} MB)")
        return False

    wanted = with_embeddings
    if not wanted and interactive:
        out("")
        wanted = prompt_yn(offer_text(), False)
    if not wanted:
        out("  Semantic search: off (word matching only).")
        out(manual_instructions())
        return False

    out(f"  Running: {' '.join(pip_command())}")
    ok, detail = _install_fastembed(run)
    if not ok:
        out(f"  Could not install fastembed: {detail}")
        out("  Continuing with word matching only.")
        out(manual_instructions())
        return False

    out(f"  Downloading the model from huggingface.co ({MODEL_DOWNLOAD_MB} MB, once)...")
    ok, detail = _download_model()
    if not ok:
        out(f"  Could not download the model: {detail}")
        out("  Continuing with word matching only. It will be fetched on the first search that needs it.")
        return False

    out("  Semantic search: on (fastembed). Runs on this machine.")
    out(f"  Indexing with embeddings takes about {SECONDS_PER_1000_SECTIONS} seconds per 1,000 sections, once.")
    return True

def _offer_repair(
    problem: str,
    *,
    interactive: bool,
    with_embeddings: bool,
    dry_run: bool,
    prompt_yn: Callable[[str, bool], bool],
    out: Callable[[str], None],
) -> bool:
    """FastEmbed is installed and its model won't load: report it, repair on consent.

    Same consent rule as the install: ``--with-embeddings`` or a yes at the
    prompt, never ``--yes``. Returns True only when the model now loads.
    """
    if dry_run:
        out(f"  Semantic search: configured (fastembed), but its model won't load: {problem}")
        out(f"  With --with-embeddings (or a yes at the prompt) init would download the model ({MODEL_DOWNLOAD_MB} MB).")
        return False
    wanted = with_embeddings
    if not wanted and interactive:
        out("")
        wanted = prompt_yn(repair_text(problem), False)
    if not wanted:
        out(f"  Semantic search: configured (fastembed), but its model won't load: {problem}")
        out("  Searches match words only. To repair: jdocmunch-mcp init --with-embeddings")
        return False
    out(f"  Downloading the model from huggingface.co ({MODEL_DOWNLOAD_MB} MB, once)...")
    ok, detail = _download_model()
    if not ok:
        out(f"  Could not download the model: {detail}")
        out("  Continuing with word matching only.")
        return False
    problem = _model_problem()
    if problem:
        out(f"  The model downloaded but still won't load: {problem}")
        out("  Continuing with word matching only.")
        return False
    out("  Semantic search: on (fastembed). Model repaired. Runs on this machine.")
    return True
