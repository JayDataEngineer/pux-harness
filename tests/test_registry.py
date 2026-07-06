"""The single-source-of-truth registry is actually single-source.

These tests exist to make the manual-sync hazard (the original drift) a
**machine-enforced** failure: every name set the contract validates against and
every builder ``graph.py`` calls must DERIVE from ``REGISTRY``, with no second
hand-maintained copy anywhere. Adding a ``ToolSpec`` line is the only way to
land a new tool — if a frozenset or builder ever drifts out of sync with the
registry, one of these assertions fires.

Imports go through the package public (``pux_harness.sandbox.tools``) so the
``__init__`` re-export surface is exercised too.
"""

from enum import Enum

from pux_harness.sandbox.tools import (
    REGISTRY,
    Category,
    GRADER_TOOL_NAMES,
    LEGACY_TOOL_NAMES,
    NATIVE_FS_TOOLS,
    SPECIALIST_TOOL_NAMES,
    SPECIALISTS,
    ToolSpec,
    build_grader_tools,
    build_native_specialists,
    classify_slug,
    prefixed,
)
from pux_harness.sandbox.tools._shared import PUX_GRADER_PREFIX, PUX_PREFIX

# A sentinel exec_client — factories bind it at build time and only USE it at
# tool-invocation time, so any object builds fine (mirrors test_browser_tools).
_EXEC = "DUMMY-EXEC"


# --- REGISTRY well-formedness ----------------------------------------------


def test_registry_is_non_empty_and_partitioned():
    """50 tools: 40 specialist + 7 native + 3 grader. Catches an accidental
    add/drop in any one partition. (Phase 19 added 7 browser specialists:
    drag/hover/press/click_at/scroll_into_view/a11y/iframe.)"""
    counts = {c: 0 for c in Category}
    for spec in REGISTRY:
        assert isinstance(spec, ToolSpec)
        counts[spec.category] += 1
        assert spec.slug, f"empty slug in {spec!r}"
    assert len(REGISTRY) == 50, counts
    assert counts == {
        Category.SPECIALIST: 40,
        Category.NATIVE: 7,
        Category.GRADER: 3,
    }, counts


def test_native_specs_have_no_factory_others_do():
    """NATIVE entries are middleware-provided (factory=None); every SPECIALIST
    + GRADER entry must carry a callable factory, or build_tools silently
    drops it."""
    for spec in REGISTRY:
        if spec.category is Category.NATIVE:
            assert spec.factory is None, f"{spec.slug}: native must have no factory"
        else:
            assert callable(spec.factory), (
                f"{spec.slug}: {spec.category.name} must declare a factory"
            )


def test_category_is_plain_enum_not_strenum():
    """``Category`` must be a plain ``Enum`` so a category never compares equal
    to a bare slug string. If someone flips it to ``StrEnum``,
    ``Category.NATIVE == 'native'`` becomes True and ``classify_slug`` /
    whitelist comparisons start matching the wrong things."""
    assert issubclass(Category, Enum)
    assert Category.NATIVE != "native"
    assert Category.SPECIALIST != "specialist"
    assert Category.GRADER != "grader"


# --- derived sets track REGISTRY exactly -----------------------------------


def test_specialist_name_sets_derive_from_registry():
    """SPECIALISTS (bare) and SPECIALIST_TOOL_NAMES (prefixed) are exactly the
    SPECIALIST partition of REGISTRY — no more, no less."""
    bare = {s.slug for s in REGISTRY if s.category is Category.SPECIALIST}
    assert SPECIALISTS == bare
    assert SPECIALIST_TOOL_NAMES == {PUX_PREFIX + s for s in bare}


def test_native_fs_tools_derive_from_registry():
    """NATIVE_FS_TOOLS is exactly the NATIVE partition. Also re-asserts the
    known FilesystemMiddleware surface (the contract test at
    test_org_contract.py:109 asserts the same literal — both must agree)."""
    bare = {s.slug for s in REGISTRY if s.category is Category.NATIVE}
    assert NATIVE_FS_TOOLS == bare
    assert NATIVE_FS_TOOLS == {
        "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
    }


def test_grader_name_sets_derive_from_registry():
    """GRADERS (bare) and GRADER_TOOL_NAMES (prefixed) are exactly the GRADER
    partition — even though bare grader slugs COLLIDE with native ones
    (execute/read_file/grep exist in both), filtering by category first keeps
    both sets well-defined."""
    bare = {s.slug for s in REGISTRY if s.category is Category.GRADER}
    assert GRADERS_FROM_NAMES() == bare
    assert GRADER_TOOL_NAMES == {PUX_GRADER_PREFIX + s for s in bare}


def GRADERS_FROM_NAMES():
    """GRADERS is not re-exported through the package public (only the
    prefixed GRADER_TOOL_NAMES is); reconstruct the bare set by un-prefixing
    so this test doesn't reach into a private name."""
    return {n[len(PUX_GRADER_PREFIX):] for n in GRADER_TOOL_NAMES}


# --- the derivation proof (the whole point) --------------------------------


def test_build_native_specialists_matches_specialist_tool_names():
    """The builder returns EXACTLY SPECIALIST_TOOL_NAMES — the manual-sync
    hazard between the old 33-call factory list and the frozenset is now
    machine-enforced. A new ToolSpec that the builder forgets to instantiate
    fails here."""
    built = {t.name for t in build_native_specialists(exec_client=_EXEC)}
    assert built == SPECIALIST_TOOL_NAMES, (
        f"drift: built-only={built - SPECIALIST_TOOL_NAMES} "
        f"declared-only={SPECIALIST_TOOL_NAMES - built}"
    )


def test_build_grader_tools_matches_grader_tool_names():
    """Symmetric derivation proof for the grader surface."""
    built = {t.name for t in build_grader_tools(exec_client=_EXEC)}
    assert built == GRADER_TOOL_NAMES, (
        f"drift: built-only={built - GRADER_TOOL_NAMES} "
        f"declared-only={GRADER_TOOL_NAMES - built}"
    )


# --- the shared classifier (contract + runtime share it) --------------------


def test_classify_slug_native_wins_over_grader_collision():
    """``execute``/``read_file``/``grep`` exist in BOTH native and grader.
    ``classify_slug`` checks NATIVE first — correct, because graders never
    appear in an agent whitelist (an agent asking for ``read_file`` means the
    native fs tool, not the grader evidence tool)."""
    assert classify_slug("ls") is Category.NATIVE
    assert classify_slug("read_file") is Category.NATIVE  # NOT grader
    assert classify_slug("execute") is Category.NATIVE    # NOT grader
    assert classify_slug("grep") is Category.NATIVE       # NOT grader
    assert classify_slug("write_file") is Category.NATIVE
    assert classify_slug("edit_file") is Category.NATIVE
    assert classify_slug("glob") is Category.NATIVE


def test_classify_slug_specialist_and_unknown():
    assert classify_slug("python") is Category.SPECIALIST
    assert classify_slug("browser_navigate") is Category.SPECIALIST
    assert classify_slug("describe_image") is Category.SPECIALIST
    # a stale / typo'd reference resolves to nothing — the contract flags it,
    # the runtime raises KeyError. Both paths share this classifier.
    assert classify_slug("totally_made_up") is None
    assert classify_slug("bash") is None  # legacy name, deliberately absent


def test_classify_slug_covers_every_specialist_and_native():
    """Every declared specialist + native slug classifies (no declared tool
    is a None); only unknowns return None."""
    for slug in SPECIALISTS:
        assert classify_slug(slug) is Category.SPECIALIST, slug
    for slug in NATIVE_FS_TOOLS:
        assert classify_slug(slug) is Category.NATIVE, slug


# --- prefixed() — the one place prefix→category lives ----------------------


def test_prefixed_maps_category_to_prefix():
    assert prefixed("python", Category.SPECIALIST) == "pux_sandbox_python"
    assert prefixed("execute", Category.GRADER) == "pux_grader_execute"
    # NATIVE has no prefix (bare slug, middleware-provided).
    assert prefixed("ls", Category.NATIVE) == "ls"
    assert prefixed("read_file", Category.NATIVE) == "read_file"
    # the prefixed names actually appear in the derived sets.
    assert prefixed("python", Category.SPECIALIST) in SPECIALIST_TOOL_NAMES
    assert prefixed("execute", Category.GRADER) in GRADER_TOOL_NAMES


# --- the legacy denylist (permanent tripwire) ------------------------------


def test_legacy_tool_names_is_non_empty_denylist():
    """The frozen ``pux_sandbox_bash`` / ``pux_sandbox_file_*`` surface the
    native flip replaced. A denylist (deliberately absent from REGISTRY) so
    the dev-bot forcing surface check (main.py) imports one constant instead
    of re-declaring the literal."""
    assert LEGACY_TOOL_NAMES == {
        "pux_sandbox_bash", "pux_sandbox_file_read", "pux_sandbox_file_write",
        "pux_sandbox_file_edit", "pux_sandbox_file_glob", "pux_sandbox_file_grep",
    }


def test_legacy_names_are_not_re_introduced_as_specialists():
    """No legacy name sneaks back into the live specialist surface — that
    would silently re-expose a tool the native flip deliberately killed
    (no-legacy-left-behind)."""
    assert LEGACY_TOOL_NAMES.isdisjoint(SPECIALIST_TOOL_NAMES)
    assert LEGACY_TOOL_NAMES.isdisjoint(NATIVE_FS_TOOLS)
