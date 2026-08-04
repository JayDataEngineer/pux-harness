"""Inheritance-chain introspection — the renderer behind ``pux org chain``.

Prints the extends-chain (root→child), which files each org in the chain
contributes, and the per-file merge rule that applies. Read-only — no behavior
change, pure introspection so experimenters can see exactly where each piece of
an org's prompt/config/policy comes from WITHOUT reading the loader source.

The merge rules are FIXED per file type (they exist because the files serve
different purposes: AGENTS.md is free-text prompt prose → concatenation;
profile.yaml is config → deep-merge; policy.yaml is sandbox policy →
never inherited; etc.). This module makes those rules DISCOVERABLE at the
point of inspection instead of buried in loader docstrings.
"""
from __future__ import annotations

from pathlib import Path


# The per-file merge rules. These are FIXED (determined by the file's purpose),
# not computed — this table is the SINGLE SOURCE of truth for the labels.
_MERGE_RULES: dict[str, str] = {
    "AGENTS.md": "concatenation (root→child; each ancestor's body appended, newline-joined)",
    "profile.yaml": "deep-merge (root→child; scalars: child wins; dicts: merged recursively)",
    "policy.yaml": "never inherited (each org owns its own sandbox policy)",
    "org.yaml": "extends + inherit_roster (the `extends:` field IS the chain; "
                "`inherit_roster: false` opts out of the parent's agent roster)",
    "agents/*.md": "extends-merge (frontmatter: delta-wins per field; body: "
                   "base + delta concatenated with blank line)",
}

# Which files are relevant per org. Checked for presence in each chain member.
_ORG_FILES = ("AGENTS.md", "profile.yaml", "policy.yaml", "org.yaml")


def _resolve_org_dir(name: str, project_root: Path) -> Path | None:
    """Resolve an org name to its directory, or None if not found."""
    from pux_harness.kit._paths import search_org_dir

    try:
        return search_org_dir(name, project_root)
    except FileNotFoundError:
        return None


def _check_yaml_key(org_dir: Path, filename: str, key: str) -> str | None:
    """Read a top-level key from a YAML file, or None if absent/unreadable."""
    import yaml

    path = org_dir / filename
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    val = data.get(key)
    return str(val) if val is not None else None


def render_org_chain(org: str, project_root: Path) -> str:
    """The full inheritance report for ``org``.

    Sections:
    1. The extends-chain (root→child with arrows, plus each org's extends target)
    2. Files per org in the chain (present/absent, with a note on what it does)
    3. Merge rules by file type (the FIXED table)
    4. Effective supervisor base composition (which files contribute to the
       agents_md_core part)
    """
    from pux_harness.kit.loaders import _resolved_org_chain

    # Fail loud if the target org itself doesn't resolve (the CLI catches this
    # and prints a clean error). _resolved_org_chain falls back to [name] on a
    # broken chain, so we check the org dir directly.
    target_dir = _resolve_org_dir(org, project_root)
    if target_dir is None:
        msg = f"org {org!r} not found under orgs/specialists/{org}/, orgs/{org}/, or orgs/_shared/"
        raise FileNotFoundError(msg)

    chain = _resolved_org_chain(org, project_root)  # root→child

    # --- Section 1: the chain ---
    lines: list[str] = [f"=== Inheritance chain for {org!r} (root→child) ===", ""]
    if len(chain) == 1:
        lines.append(f"  {chain[0]}  (no extends — standalone org)")
    else:
        for i, name in enumerate(chain):
            org_dir = _resolve_org_dir(name, project_root)
            extends = (
                _check_yaml_key(org_dir, "org.yaml", "extends") if org_dir else None
            )
            arrow = "  " if i == 0 else "→ "
            ext_note = f"  (extends: {extends})" if extends else ""
            role = "base org" if i == 0 else ("★ target" if i == len(chain) - 1 else "ancestor")
            if i == len(chain) - 1 and i > 0:
                role = "specialist (this org)"
            lines.append(f"  {arrow}{name}  [{role}]{ext_note}")
    lines.append("")

    # --- Section 2: files per org ---
    lines.append("=== Files per org in chain ===")
    for name in chain:
        org_dir = _resolve_org_dir(name, project_root)
        if org_dir is None:
            lines.append(f"  {name}/  (dir not found)")
            lines.append("")
            continue
        rel = org_dir.relative_to(project_root) if org_dir.is_relative_to(project_root) else org_dir
        lines.append(f"  {rel}/")
        for fname in _ORG_FILES:
            fpath = org_dir / fname
            if fpath.is_file():
                lines.append(f"    {fname:<16} ✅ present")
            else:
                note = "(absent — no org-wide suffix/rubric/middleware)" if fname == "profile.yaml" else "(absent)"
                lines.append(f"    {fname:<16} —  {note}")
        # Show agents/ dir
        agents_dir = org_dir / "agents"
        if agents_dir.is_dir():
            count = sum(1 for _ in agents_dir.glob("*.md"))
            lines.append(f"    agents/           ✅ {count} agent(s)")
        lines.append("")

    # --- Section 3: merge rules ---
    lines.append("=== Merge rules by file type (FIXED — same for every org) ===")
    for fname, rule in _MERGE_RULES.items():
        lines.append(f"  {fname:<16} {rule}")
    lines.append("")

    # --- Section 4: effective supervisor base ---
    lines.append("=== Effective supervisor base (agents_md_core part) ===")
    parts: list[str] = []
    for name in chain:
        org_dir = _resolve_org_dir(name, project_root)
        if org_dir and (org_dir / "AGENTS.md").is_file():
            rel = org_dir.relative_to(project_root) if org_dir.is_relative_to(project_root) else org_dir
            parts.append(str(rel / "AGENTS.md"))
    addendum_path = project_root / "orgs" / "_shared" / "harness_addendum.md"
    if addendum_path.is_file():
        parts.append("orgs/_shared/harness_addendum.md")
    else:
        parts.append("(embedded _ADDENDUM fallback — orgs/_shared/harness_addendum.md absent)")
    for i, p in enumerate(parts):
        prefix = "  " if i == 0 else "+ "
        lines.append(f"  {prefix}{p}")
    lines.append("")

    return "\n".join(lines)
