"""TUI launcher — bridges ``pux tui --org <name>`` to ``dcode -a <name>``.

Production architecture
-----------------------
dcode (deepagents-code) is a self-contained, Claude-Code-equivalent TUI. It works
by spawning a ``langgraph dev`` subprocess from its OWN uv-tool venv — and that
subprocess needs ``langgraph_cli`` + ``langgraph_api``, which live ONLY in dcode's
venv, NOT in the harness interpreter. Running dcode in-process under the harness
python therefore crashes with "Server process exited with code 1"
(``sys.executable`` lacks ``langgraph_cli``).

So this launcher runs dcode under **dcode's own interpreter**:

1. Ensures dcode is installed (one-time ``uv tool install`` bootstrap).
2. If the current process is NOT dcode's python, ``os.execvpe`` re-execs into it
   with the harness dir on ``PYTHONPATH`` so ``pux_harness`` stays importable.
   After re-exec we land back in ``main()`` → ``run_tui()`` → in-process path.
3. Under dcode's python: install the org persona into ``~/.deepagents/<org>/``
   (CTO ``AGENTS.md`` + the portable subagents), monkey-patch the P-U-X banner
   and set the org subheader, then call ``deepagents_code.main.cli_main()``.

The banner patch is the ONE thing that demands in-process execution: dcode
exposes no env-var override for banner text (only for the subheader). Everything
else — org→agent wiring, subheader, model, subagents — uses dcode's native,
supported configuration surfaces.

Org → agent mapping uses dcode's native model: ``dcode -a <org>`` reads
``~/.deepagents/<org>/AGENTS.md``. The project-root ``.deepagents/`` build from
the previous wrapper resolved the wrong path and is gone.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from pux_harness.kit._paths import project_root
from pux_harness.tui_branding import get_branding, get_pux_banner

# HARNESS_ROOT = the dir containing the ``pux_harness/`` package. dcode's
# interpreter needs it on PYTHONPATH so ``pux_harness`` stays importable after
# the re-exec. Derived from ``__file__`` (NOT the app root) so it's correct
# wherever the harness is installed — the orchestrator layout today, the
# standalone pux-harness repo after the split.
HARNESS_ROOT = Path(__file__).resolve().parents[1]
# PROJECT_ROOT = the APP root (``orgs/`` + ``.pux/`` + ``AGENTS.md``), injected
# via the kit's resolver — NOT the install path. ``bin/pux`` exports
# ``PUX_PROJECT_ROOT=$REPO`` before exec, so this resolves to the orchestrator
# root and the TUI finds the orgs.
PROJECT_ROOT = project_root()

# dcode uv-tool venv layout.
DCODE_VENV = Path.home() / ".local" / "share" / "uv" / "tools" / "deepagents-code"
DCODE_PYTHON = DCODE_VENV / "bin" / "python"
DCODE_CLI = DCODE_VENV / "bin" / "dcode"

# Subagents ported from the org into dcode's `task` delegation surface.
# `web-agent` is intentionally excluded: it drives pux-only browser tools
# (browser_navigate/click/..., Phase 16/20) absent in dcode — porting it would
# need a browser MCP. See ``_adapt_cto_prompt`` for how its clause is reworked.
_PORTED_SUBAGENTS = ("code-worker", "dev-bot-explorer")


# ---------------------------------------------------------------------------
# dcode install / interpreter resolution
# ---------------------------------------------------------------------------

def _dcode_present() -> bool:
    """True if the dcode uv-tool venv + entry point both exist."""
    return DCODE_PYTHON.is_file() and (DCODE_CLI.is_file() or bool(shutil.which("dcode")))


def _ensure_dcode_installed() -> Path:
    """Return dcode venv python, bootstrapping dcode via uv if absent.

    This is the "easy install" path: if dcode is missing, install it once with
    ``uv tool install deepagents-code`` (the same command the official installer
    runs). Idempotent — a present install short-circuits.
    """
    if _dcode_present():
        return DCODE_PYTHON
    print("pux tui: dcode not found — installing via uv (one-time bootstrap)…",
          file=sys.stderr)
    uv = shutil.which("uv")
    if not uv:
        print("pux tui: 'uv' not on PATH. Install it from https://astral.sh/uv "
              "(curl -LsSf https://astral.sh/uv/install.sh | sh) and re-run.",
              file=sys.stderr)
        raise SystemExit(1)
    try:
        subprocess.run([uv, "tool", "install", "deepagents-code"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"pux tui: dcode install failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not _dcode_present():
        print("pux tui: install reported success but the dcode venv is missing.",
              file=sys.stderr)
        raise SystemExit(1)
    return DCODE_PYTHON


def _running_under_dcode(dcode_py: Path) -> bool:
    """True if this process is already running under dcode's interpreter."""
    try:
        return Path(sys.executable).resolve() == dcode_py.resolve()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Org resolution (mirrors agent/orgs.py discovery)
# ---------------------------------------------------------------------------

def _discover_orgs() -> list[str]:
    """Sorted names of every org dir containing ``AGENTS.md``."""
    orgs: list[str] = []
    for root in [PROJECT_ROOT / "orgs", PROJECT_ROOT / "orgs" / "specialists"]:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "AGENTS.md").is_file():
                orgs.append(child.name)
    return sorted(orgs)


def _resolve_org(name: str) -> Path:
    """Return the org dir for *name* (``orgs/`` then ``orgs/specialists/``)."""
    for cand in [PROJECT_ROOT / "orgs" / name,
                 PROJECT_ROOT / "orgs" / "specialists" / name]:
        if cand.is_dir() and (cand / "AGENTS.md").is_file():
            return cand
    print(f"pux tui: unknown org {name!r}. Available: {', '.join(_discover_orgs())}",
          file=sys.stderr)
    raise SystemExit(1)


def _resolve_agent_md(org_dir: Path, slug: str) -> Path | None:
    """Find ``<slug>.md`` — org-local first, then ``_shared``."""
    local = org_dir / "agents" / f"{slug}.md"
    if local.is_file():
        return local
    shared = PROJECT_ROOT / "orgs" / "_shared" / "agents" / f"{slug}.md"
    return shared if shared.is_file() else None


# ---------------------------------------------------------------------------
# Persona adaptation: pux org prompt → dcode-native prompt
# ---------------------------------------------------------------------------

def _adapt_cto_prompt(text: str) -> str:
    """Adapt the canonical CTO AGENTS.md to dcode's tool surface.

    Targeted transforms (the methodology — PLAN/EXECUTE/RECOVER/ESCALATE, risk
    tiers, verify-or-die, ship gate, delegation to code-worker +
    dev-bot-explorer — is unchanged; dcode's tool names already match):

      * Drop the ``pux_sandbox_python`` bullet (no such tool in dcode; the
        ``execute`` tool covers python one-liners).
      * Rework the live-browser delegation to ``web-agent`` — that specialist
        drives pux-only browser tools absent here, so verify inline via
        ``execute``/``fetch_url`` instead.
    """
    # Drop the pux_sandbox_python bullet and its continuation line(s).
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if "pux_sandbox_python" in lines[i]:
            i += 1
            # consume indented continuation lines (not a new bullet/header/blank)
            while (i < len(lines)
                   and lines[i].startswith(" ")
                   and not re.match(r"\s*(-|\*|`|#)", lines[i])):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    text = "\n".join(out)

    # web-agent delegation clause (Verify section) → inline verification.
    text = re.sub(
        r"If the deliverable is a web site, delegate the live-?\s*browser "
        r"checks to `web-agent`.*?findings\.",
        "If the deliverable is a web site, verify it inline (curl/playwright via "
        "`execute`, or `fetch_url`) — a live-browser specialist is not available "
        "in TUI mode",
        text, flags=re.DOTALL,
    )
    # web-agent mention in the intro delegation sentence → drop (subagent not ported).
    text = re.sub(
        r", and live-?\s*browser e2e verification to `web-agent`",
        "",
        text, flags=re.DOTALL,
    )
    return text


def _adapt_subagent_prompt(text: str) -> str:
    """Adapt a subagent ``.md`` to dcode: drop pux-only refs, fix workspace path.

    dcode ignores the frontmatter ``tools:`` field (it reads only name /
    description / model), so the pux tool whitelist is left in place harmlessly.
    """
    # pux_sandbox_python → execute (token-level; dcode's `execute` runs python).
    text = text.replace("pux_sandbox_python", "execute")
    # Drop the FilesystemMiddleware provenance sentence (multi-line tolerant).
    text = re.sub(
        r"\s*These are always available to you regardless of the\s*"
        r"`tools:` whitelist \(they come from `FilesystemMiddleware`\)\.",
        "", text, flags=re.DOTALL,
    )
    # /sandbox/workspace/ container sentence → current working directory.
    text = re.sub(
        r"The workspace lives at `/sandbox/workspace/` inside the sandbox "
        r"container[^\n]*\n\s*that'?s the project root\.",
        "The workspace is your current working directory (the project root).",
        text, flags=re.DOTALL,
    )
    text = text.replace("/sandbox/workspace/", "./")
    return text


# ---------------------------------------------------------------------------
# Persona install → ~/.deepagents/<org>/
# ---------------------------------------------------------------------------

def _install_agent(org: str, org_dir: Path) -> Path:
    """Install/refresh ``~/.deepagents/<org>/`` from the org's canonical source.

    Writes the adapted CTO ``AGENTS.md`` and ports the supported subagents to
    ``~/.deepagents/<org>/agents/<name>/AGENTS.md`` (where dcode's
    ``load_async_subagents`` discovers them for the ``task`` tool).

    Idempotent. The org source under ``orgs/`` is the source of truth — manual
    edits to the installed copy are overwritten on the next launch.
    """
    target = Path.home() / ".deepagents" / org
    target.mkdir(parents=True, exist_ok=True)

    cto = (org_dir / "AGENTS.md").read_text(encoding="utf-8")
    (target / "AGENTS.md").write_text(_adapt_cto_prompt(cto), encoding="utf-8")

    agents_dir = target / "agents"
    for slug in _PORTED_SUBAGENTS:
        src = _resolve_agent_md(org_dir, slug)
        if src is None:
            print(f"pux tui: warning: subagent {slug!r} not found, skipping",
                  file=sys.stderr)
            continue
        body = src.read_text(encoding="utf-8")
        sa_dir = agents_dir / slug
        sa_dir.mkdir(parents=True, exist_ok=True)
        (sa_dir / "AGENTS.md").write_text(_adapt_subagent_prompt(body), encoding="utf-8")

    return target


# ---------------------------------------------------------------------------
# Banner monkey-patching (in-process only)
# ---------------------------------------------------------------------------

def _patch_banner(org: str) -> None:
    """Monkey-patch dcode's banner to P-U-X and set the org subheader.

    Must run under dcode's python, before ``cli_main()`` mounts the welcome
    screen. dcode exposes no env-var banner override, so in-process patching is
    the only path. The subheader DOES have an env var
    (``DEEPAGENTS_CODE_DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER``) and survives the
    langgraph subprocess boundary.
    """
    import deepagents_code.config as _cfg  # noqa: PLC0415

    banner = get_pux_banner()
    # Patch the constants (in case any caller reads them directly) AND the
    # function (the path welcome.py actually calls).
    _cfg._UNICODE_BANNER = banner
    _cfg._ASCII_BANNER = banner
    _cfg.get_banner = lambda: banner  # type: ignore[assignment]
    os.environ["DEEPAGENTS_CODE_DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER"] = (
        get_branding(org)["subheader"]
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def _run_in_process(
    org: str,
    task: str | None,
    model: str | None,
    auto_approve: bool,
    shell_allow: str | None,
) -> None:
    """Install persona, patch banner, and call dcode's cli_main in-process."""
    org_dir = _resolve_org(org)
    rel = org_dir.relative_to(PROJECT_ROOT)
    print(f"pux tui: installing ~/.deepagents/{org}/ from {rel}", file=sys.stderr)
    _install_agent(org, org_dir)
    _patch_banner(org)

    argv = ["dcode", "-a", org]
    if model:
        argv += ["-M", model]
    if task is not None:
        # Headless: -n runs one task and exits. Auto-approve file ops (-y) and
        # allow a safe shell set (-S) so the agent can actually act unattended.
        argv += ["-n", task, "-y", "-S", (shell_allow or "recommended")]
    else:
        # Interactive TUI: the user approves via Shift+Tab / prompts.
        if auto_approve:
            argv += ["-y"]
        if shell_allow:
            argv += ["-S", shell_allow]
    sys.argv = argv

    print(f"pux tui: launching: {' '.join(argv)}", file=sys.stderr)
    from deepagents_code.main import cli_main  # noqa: PLC0415
    cli_main()


def _reexec(
    dcode_py: Path,
    org: str,
    task: str | None,
    model: str | None,
    auto_approve: bool,
    shell_allow: str | None,
) -> None:
    """Re-exec this launcher under dcode's python (no return)."""
    env = os.environ.copy()
    # Keep pux_harness importable once we're in dcode's venv.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{HARNESS_ROOT}{os.pathsep}{existing}" if existing else str(HARNESS_ROOT)
    )
    argv = [str(dcode_py), "-m", "pux_harness.tui", "--org", org]
    if task is not None:
        argv += ["--task", task]
    if model:
        argv += ["--model", model]
    if auto_approve:
        argv += ["--auto-approve"]
    if shell_allow:
        argv += ["--shell-allow", shell_allow]
    os.execvpe(str(dcode_py), argv, env)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_orgs() -> None:
    """Print discovered orgs and exit."""
    orgs = _discover_orgs()
    if not orgs:
        print("No orgs found.")
        return
    print(f"{len(orgs)} orgs:")
    for name in orgs:
        org_dir = _resolve_org(name)
        agent_count = 0
        org_yaml = org_dir / "org.yaml"
        if org_yaml.is_file():
            with open(org_yaml) as f:
                agent_count = len((yaml.safe_load(f) or {}).get("agents", []))
        suffix = f"  ({agent_count} agents)" if agent_count else ""
        print(f"  {name}{suffix}")


def run_tui(
    org: str,
    task: str | None = None,
    model: str | None = None,
    auto_approve: bool = False,
    shell_allow: str | None = None,
) -> None:
    """Launch dcode with *org*'s persona + branding.

    With *task*: headless one-shot (auto-approve + safe shell on). Without: the
    interactive TUI. Re-execs under dcode's python first if not already there.
    """
    dcode_py = _ensure_dcode_installed()
    if _running_under_dcode(dcode_py):
        _run_in_process(org, task, model, auto_approve, shell_allow)
    else:
        print(f"pux tui: re-execing under dcode python ({dcode_py})", file=sys.stderr)
        _reexec(dcode_py, org, task, model, auto_approve, shell_allow)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m pux_harness.tui`` (used after re-exec)."""
    ap = argparse.ArgumentParser(
        prog="pux tui",
        description="Launch dcode TUI with an org persona + Pux branding.",
    )
    ap.add_argument("--org", default=os.environ.get("PUX_ORG", "general"))
    ap.add_argument("--task", default=None,
                    help="headless task — runs non-interactively and exits")
    ap.add_argument("--model", default=None,
                    help="dcode model id (provider:model), e.g. opencode-go-openai:kimi-k2.7-code")
    ap.add_argument("--auto-approve", action="store_true",
                    help="auto-approve tool calls (dcode -y)")
    ap.add_argument("--shell-allow", default=None,
                    help="shell allow-list for headless/interactive (dcode -S)")
    ap.add_argument("--list-orgs", action="store_true", help="list orgs and exit")
    args = ap.parse_args(argv)

    if args.list_orgs:
        list_orgs()
        return
    run_tui(args.org, args.task, args.model, args.auto_approve, args.shell_allow)


if __name__ == "__main__":
    main()
