"""OCI artifact emission via the ``oras`` CLI (dynamic-tools P5).

A pack's ``.tar.gz`` is a flat archive; the OCI artifact is its content-addressed,
tamper-evident, registry-pushable form. Built by **shelling out to ``oras``**
(Decision 2 — reuse-first: ``oras`` handles digests / manifest / ``oci-layout`` /
``index.json``; we do NOT hand-roll blobs). This is the thin pux glue
([[rely-on-upstream]]) over a mature CNCF tool.

Layered so the mutable agent-library (``lib/``) gets its OWN digest — the integrity
target. The design's §2.3 model:

  - ``config``        → a compact descriptor, mediaType
                        ``application/vnd.pux.org.config.v1+json`` (``oras --config``).
  - ``source-code``   → primitives (non-``lib``) + the runtime scaffold
                        (kit / run.py / pyproject / README), mediaType
                        ``application/vnd.pux.org.layer.source-code.v1.tar``.
  - ``agent-library`` → ``orgs/<org>/lib/**`` (the LEARNED functions + index.yaml),
                        mediaType ``application/vnd.pux.org.layer.agent-library.v1.tar``.
                        Its digest is the tamper anchor — flip a byte in a learned
                        function and this layer's SHA-256 changes.

``oras push --oci-layout <dir>:<tag>`` writes a LOCAL OCI image-layout directory
(no registry needed for the P5 proof); ``--export-manifest`` + ``--format json``
capture the manifest + layer digests → ``provenance.json``. The output is consumable
by ``oras``/``crane``/``skypeo`` and pushable to GHCR/Docker Hub later
(``oras push <registry>``) WITHOUT repackaging (Decision 3: signing is P6; this
phase reserves the slots — the layer digests make the artifact immutable +
tamper-evident *now*).

Like the gitleaks hook ([[no-fallbacks-no-aliases]]), an absent ``oras`` binary
REFUSES the emit (clear install message) — there is no silent skip.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

# --- pux OCI media types (the artifact's self-describing vocabulary) ------------
PUX_CONFIG_MEDIATYPE = "application/vnd.pux.org.config.v1+json"
PUX_SOURCE_LAYER_MEDIATYPE = "application/vnd.pux.org.layer.source-code.v1.tar"
PUX_LIBRARY_LAYER_MEDIATYPE = "application/vnd.pux.org.layer.agent-library.v1.tar"
PUX_OCI_SCHEMA = "pux-oci-v1"

# Text files whose CONTENT is normalized (``orgs/specialists/`` → ``orgs/``) so the
# unpacked tree is self-consistent — mirrored from pack.py so OCI layers match the
# tar.gz byte-for-byte (the same primitive ships in both forms).
try:  # avoid a hard import cycle; pack.py imports oci indirectly via pack_org wiring
    from pux_harness.pack import _TEXT_SUFFIXES, _normalize_specialists_refs
except Exception:  # pragma: no cover - pack always present in-tree
    _TEXT_SUFFIXES = (".md", ".yaml", ".yml", ".txt")

    def _normalize_specialists_refs(data: bytes) -> bytes:  # type: ignore[no-redef]
        return data


class OciError(Exception):
    """Raised when the OCI emit cannot proceed (e.g. ``oras`` absent)."""


@dataclass
class OciLayer:
    """One layer of the emitted artifact. ``digest`` is its SHA-256 content address."""

    type: str          # "config" | "source-code" | "agent-library"
    digest: str        # sha256:...
    media_type: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "digest": self.digest,
                "media_type": self.media_type, "size": self.size}


@dataclass
class OciArtifact:
    """The emitted artifact: its layout dir, tag, manifest digest, layers, and the
    ``provenance.json`` path that records them (the P5 audit surface)."""

    layout: Path
    tag: str
    digest: str        # the manifest digest = the artifact's content address
    media_type: str
    layers: list[OciLayer] = field(default_factory=list)
    provenance: Path | None = None

    def library_layer(self) -> OciLayer | None:
        """The ``agent-library`` layer — the integrity anchor (mutable ``lib/``)."""
        return next((layer for layer in self.layers if layer.type == "agent-library"), None)


# ---------------------------------------------------------------------------
# Layer grouping + tar construction
# ---------------------------------------------------------------------------

def _is_library(path: str, org: str) -> bool:
    """``orgs/<org>/lib/**`` — the agent-authored learned functions + index.yaml.

    Anchored on ``/<org>/lib/`` (not a bare ``/lib/`` substring) so a sandbox
    ``lib/`` subdir (``orgs/<org>/sandbox/lib/util.py``) is NOT misrouted into the
    agent-library integrity layer. Handles both the flat and the specialists
    (``orgs/specialists/<org>/lib/``) layouts — the org segment disambiguates
    regardless of the ``specialists/`` prefix."""
    return f"/{org}/lib/" in path


def _normalized_bytes(archive_path: str, raw: bytes) -> bytes:
    """Apply the same text-flattening the tar.gz uses so both forms ship identically."""
    if archive_path.endswith(_TEXT_SUFFIXES):
        return _normalize_specialists_refs(raw)
    return raw


def _split_layers(
    files: dict[str, Path], scaffold: dict[str, bytes], org: str,
) -> tuple[list[tuple[str, bytes]], list[tuple[str, bytes]]]:
    """Group the collected content into (source-code entries, agent-library entries).

    ``files`` are host-Path primitives; ``scaffold`` are generated bytes (kit + run.py
    + pyproject + README). The scaffold is always source-code (it is the trusted
    portable compiler, never the agent's learned library)."""
    source: list[tuple[str, bytes]] = []
    library: list[tuple[str, bytes]] = []
    for archive_path, host_path in sorted(files.items()):
        try:
            raw = host_path.read_bytes()
        except (PermissionError, OSError):
            continue  # mirror pack_org: unreadable files are skipped, not fatal
        data = _normalized_bytes(archive_path, raw)
        (library if _is_library(archive_path, org) else source).append((archive_path, data))
    # The generated scaffold joins the source layer (trusted kit, not learned lib).
    for archive_path, raw in sorted(scaffold.items()):
        source.append((archive_path, _normalized_bytes(archive_path, raw)))
    return source, library


def _build_layer_tar(entries: list[tuple[str, bytes]], prefix: str) -> bytes:
    """An UNCOMPRESSED tar holding ``<prefix>/<archive_path>`` for each entry.
    Uncompressed so the SHA-256 is a clean content hash (no gzip mtime/OS nonce).
    Empty entries → empty tar (oras still records the layer; its digest is stable)."""
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for archive_path, data in entries:
            info = tarfile.TarInfo(name=f"{prefix}/{archive_path}")
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# oras subprocess
# ---------------------------------------------------------------------------

def _oras_available(runner: Callable | None = None) -> bool:
    """Is ``oras`` on PATH? (Injectable runner for offline tests — rhymes with the
    gitleaks gate.)"""
    run = runner or subprocess.run
    try:
        out = run(["oras", "version"], capture_output=True, text=True,
                  timeout=10.0, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    return getattr(out, "returncode", 1) == 0


def _run_oras_push(
    layout: Path, tag: str, *, workdir: Path, config_name: str, source_name: str,
    library_name: str, manifest_name: str, annotations: dict[str, str],
    runner: Callable | None = None, timeout: float = 180.0,
) -> tuple[int, str, str]:
    """``oras push --oci-layout <layout>:<tag>`` with the pux config + 2 layer tars +
    annotations, exporting the manifest + JSON output. Returns (code, stdout, stderr).

    The JSON ``--format`` stdout carries the manifest digest; ``--export-manifest``
    writes the full manifest (config + layer descriptors with digests).

    Run from ``workdir`` with RELATIVE layer names: oras records each file's push-time
    path as its ``org.opencontainers.image.title`` annotation, and on ``oras pull``
    it refuses to write a title outside the output dir (path-traversal guard). Pushing
    absolute temp paths would make the title an absolute path → every consumer pull
    fails with "path traversal disallowed". Relative names keep the title a clean
    basename (``agent-library.tar``), so the artifact round-trips anywhere."""
    run = runner or subprocess.run
    ref = f"{layout}:{tag}"
    cmd: list[str] = [
        "oras", "push", "--oci-layout", ref,
        "--config", f"{config_name}:{PUX_CONFIG_MEDIATYPE}",
        "--export-manifest", manifest_name,
        f"{source_name}:{PUX_SOURCE_LAYER_MEDIATYPE}",
        f"{library_name}:{PUX_LIBRARY_LAYER_MEDIATYPE}",
        "--format", "json",
    ]
    for key, val in annotations.items():
        cmd += ["--annotation", f"{key}={val}"]
    try:
        out = run(cmd, capture_output=True, text=True, timeout=timeout, check=False,
                  cwd=str(workdir))
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return 127, "", repr(exc)
    return getattr(out, "returncode", 1), getattr(out, "stdout", "") or "", \
        getattr(out, "stderr", "") or ""


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

def emit_oci_artifact(
    org: str,
    files: dict[str, Path],
    scaffold: dict[str, bytes],
    inventory: dict[str, Any],
    output_layout: Path | None = None,
    *,
    tag: str | None = None,
    oras_runner: Callable | None = None,
    provenance_name: str = "provenance.json",
) -> OciArtifact:
    """Emit the layered OCI artifact for an org's collected pack content.

    Reuses the SAME ``files`` + ``scaffold`` + ``inventory`` ``pack_org`` already
    computes (additive — the ``.tar.gz`` is unchanged). Writes a local ``oci-layout``
    directory + ``provenance.json``; returns the :class:`OciArtifact`. Raises
    :class:`OciError` if ``oras`` is absent (fail-clear; no silent skip).

    ``oras_runner`` injects the subprocess runner for deterministic offline tests
    (mirrors ``gitleaks_runner``).
    """
    if not _oras_available(runner=oras_runner):
        raise OciError(
            "oras binary not found on PATH — install it to emit the OCI artifact "
            "(dynamic-tools P5; reuse-first via the oras CLI — see oras.land). "
            "The .tar.gz pack is unaffected; OCI is an additional output.")

    import tempfile  # noqa: PLC0415 — scoped to the emit (not module import cost)

    layout = (output_layout or Path(f"{org}.oci")).resolve()
    layout.parent.mkdir(parents=True, exist_ok=True)
    artifact_tag = tag or "v1"

    source_entries, library_entries = _split_layers(files, scaffold, org)

    # Compact config descriptor (content-deterministic — no timestamp inside it; the
    # manifest digest carries oras's auto org.opencontainers.image.created, so it is
    # NOT stable across pushes, but layer digests ARE — that is what tamper-detection
    # keys on). hooks_all_ok comes from the P4 provenance block already in inventory.
    prov = inventory.get("provenance") or {}
    config_blob = json.dumps({
        "org": org,
        "schema": PUX_OCI_SCHEMA,
        "layers": {"source-code": len(source_entries),
                   "agent-library": len(library_entries)},
        "hooks_all_ok": bool(prov.get("all_ok", True)),
        "total_files": inventory.get("total_files"),
    }, indent=2, sort_keys=True).encode()

    with tempfile.TemporaryDirectory(prefix="pux-oci-") as tmp:
        tmpdir = Path(tmp)
        config_name, source_name, library_name, manifest_name = (
            "config.json", "source-code.tar", "agent-library.tar", "manifest.json")
        (tmpdir / config_name).write_bytes(config_blob)
        (tmpdir / source_name).write_bytes(_build_layer_tar(source_entries, org))
        (tmpdir / library_name).write_bytes(_build_layer_tar(library_entries, org))

        annotations = {
            "org.pux.org": org,
            "org.pux.schema": PUX_OCI_SCHEMA,
            "org.pux.layer.types": "source-code,agent-library",
        }
        code, out_json, err = _run_oras_push(
            layout, artifact_tag, workdir=tmpdir, config_name=config_name,
            source_name=source_name, library_name=library_name,
            manifest_name=manifest_name, annotations=annotations, runner=oras_runner,
        )
        if code != 0:
            raise OciError(
                f"oras push failed (exit {code}) for org {org!r}: {err.strip() or out_json}")
        try:
            pushed = json.loads(out_json)
        except json.JSONDecodeError as exc:
            raise OciError(f"oras push returned no JSON (exit {code}): {exc}; {err}") from exc
        manifest = json.loads((tmpdir / manifest_name).read_text())

    digest = pushed.get("digest") or manifest.get("config", {}).get("digest", "")
    layers = _layers_from_manifest(manifest)
    artifact = OciArtifact(
        layout=layout, tag=artifact_tag, digest=digest,
        media_type=manifest.get("mediaType", "application/vnd.oci.image.manifest.v1+json"),
        layers=layers,
    )
    artifact.provenance = _write_provenance(org, artifact, manifest, prov, layout, provenance_name)
    return artifact


def _ensure_layout(layout: Path) -> None:
    """The layout dir is created by ``oras push`` in the real flow; ensure it
    exists regardless (defensive — the injected test runner does not create it)
    so ``provenance.json`` always lands inside it."""
    layout.mkdir(parents=True, exist_ok=True)


def _layers_from_manifest(manifest: dict[str, Any]) -> list[OciLayer]:
    """The config blob + each pushed tar, mapped to pux layer types by mediaType."""
    out: list[OciLayer] = []
    cfg = manifest.get("config") or {}
    if cfg:
        out.append(OciLayer(type="config", digest=cfg.get("digest", ""),
                            media_type=cfg.get("mediaType", PUX_CONFIG_MEDIATYPE),
                            size=cfg.get("size", 0)))
    by_media = {PUX_SOURCE_LAYER_MEDIATYPE: "source-code",
                PUX_LIBRARY_LAYER_MEDIATYPE: "agent-library"}
    for layer in manifest.get("layers", []) or []:
        out.append(OciLayer(
            type=by_media.get(layer.get("mediaType", ""), "unknown"),
            digest=layer.get("digest", ""), media_type=layer.get("mediaType", ""),
            size=layer.get("size", 0)))
    return out


def _write_provenance(
    org: str, artifact: OciArtifact, manifest: dict[str, Any],
    hook_provenance: dict[str, Any], layout: Path, name: str,
) -> Path:
    """``provenance.json`` — the immutable lineage record: the artifact + layer
    digests (SHA-256) + the P4 hook results + a RESERVED (null) signing slot for P6.

    Written into the layout dir so it travels with the artifact. The library layer's
    digest is the tamper anchor — verify it against a re-derived digest to detect a
    mutated learned function."""
    _ensure_layout(layout)
    path = layout / name
    record = {
        "org": org,
        "schema": PUX_OCI_SCHEMA,
        "artifact": {
            "layout": str(layout.name),
            "tag": artifact.tag,
            "digest": artifact.digest,
            "media_type": artifact.media_type,
        },
        "layers": [layer.to_dict() for layer in artifact.layers],
        "hooks": (hook_provenance.get("hooks") or []),
        "hooks_all_ok": bool(hook_provenance.get("all_ok", True)),
        "manifest_annotations": manifest.get("annotations") or {},
        # P6 (stretch): cosign/Ed25519 signature over the manifest digest. Reserved
        # now so consumers can gate on its presence (null = unsigned, P5).
        "signature": None,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return path


# ---------------------------------------------------------------------------
# Verify (P5 close-the-loop: ``emit`` records the tamper anchor, ``verify`` checks it)
# ---------------------------------------------------------------------------

@dataclass
class VerifyCheck:
    """One named pass/fail of a verify run."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class VerifyResult:
    """Outcome of :func:`verify_oci_layout`. ``library_digest`` is the surfaced
    tamper anchor; ``ok`` is the AND of every check."""

    ok: bool
    layout: Path
    manifest_digest: str
    library_digest: str | None
    checks: list[VerifyCheck]

    def summary(self) -> str:
        lines = [f"oci layout: {self.layout}",
                 f"manifest:    {self.manifest_digest}"]
        if self.library_digest:
            lines.append(f"library:     {self.library_digest}  (tamper anchor)")
        for chk in self.checks:
            mark = "OK  " if chk.ok else "FAIL"
            suffix = f" — {chk.detail}" if chk.detail else ""
            lines.append(f"  [{mark}] {chk.name}{suffix}")
        verdict = "PASS — artifact verified" if self.ok else "FAIL — see checks above"
        lines.append(f"VERDICT: {verdict}")
        return "\n".join(lines)


def _blob_path(layout: Path, digest: str) -> Path:
    """``sha256:<hex>`` → ``<layout>/blobs/sha256/<hex>``."""
    algo, _, hexv = digest.partition(":")
    return layout / "blobs" / algo / hexv


def _read_layout_manifest(layout: Path) -> tuple[dict[str, Any], str]:
    """``index.json`` → the manifest blob it points at. Returns (manifest, digest)."""
    index = json.loads((layout / "index.json").read_text())
    manifest_digest = (index.get("manifests") or [{}])[0].get("digest", "")
    if not manifest_digest:
        raise OciError(f"{layout} index.json has no manifest digest — not a pux oci-layout")
    manifest = json.loads(_blob_path(layout, manifest_digest).read_text())
    return manifest, manifest_digest


def _library_layer_digest_from_source(
    org: str, source_root: Path | None,
) -> str | None:
    """Re-derive the agent-library layer digest from ``orgs/<org>/lib/**`` on disk,
    using the SAME construction pack uses (``_build_layer_tar`` + text normalization +
    ``_is_library`` filtering) so the comparison is byte-faithful. Returns ``None`` if
    the org has no ``lib/`` (nothing to attest). Handles both the flat
    (``orgs/<org>/lib``) and specialists (``orgs/specialists/<org>/lib``) layouts."""
    if source_root is None:
        return None
    root = Path(source_root)
    candidates = [root / "orgs" / org / "lib",
                  root / "orgs" / "specialists" / org / "lib"]
    lib_dir = next((c for c in candidates if c.is_dir()), None)
    if lib_dir is None:
        return None
    entries: list[tuple[str, bytes]] = []
    for path in sorted(lib_dir.rglob("*")):
        if not path.is_file():
            continue
        arc = path.relative_to(root).as_posix()
        if not _is_library(arc, org):  # defensive — everything under lib/ already matches
            continue
        entries.append((arc, _normalized_bytes(arc, path.read_bytes())))
    return "sha256:" + hashlib.sha256(_build_layer_tar(entries, org)).hexdigest()


def verify_oci_layout(
    layout: Path, *, org: str | None = None, source_root: Path | None = None,
    expected: str | None = None, expected_library: str | None = None,
) -> VerifyResult:
    """Verify a pux OCI layout — **stdlib-only** (no ``oras`` needed at verify time).

    A trust operation minimizes its toolchain: the layout is read directly
    (``index.json`` → manifest blob → layer blobs), so a consumer verifies an artifact
    WITHOUT the pack tool installed. Checks:

    1. **manifest integrity** — the manifest blob recomputes to the digest
       ``index.json`` references (catches a swapped manifest).
    2. **blob self-consistency** — every config/layer blob recomputes to its
       manifest-recorded digest (catches in-place corruption / bit-rot).
    3. **manifest trust anchor** (``expected``) — the manifest digest equals a
       known-good value recorded at pack time / from a signature (P6).
    4. **library trust anchor** (``expected_library``) — the agent-library layer
       digest equals a known-good value (the integrity target).
    5. **source attestation** (``org`` + ``source_root``) — re-derive the
       agent-library layer from ``orgs/<org>/lib/**`` and confirm it matches the
       packed layer (catches a learned function that drifted after the pack).

    The agent-library layer digest is surfaced as the tamper anchor — P5's headline
    property is now consumable: ``emit`` records it, ``verify`` checks it."""
    layout = Path(layout)
    checks: list[VerifyCheck] = []
    manifest, manifest_digest = _read_layout_manifest(layout)

    # 1. manifest integrity
    actual_manifest = "sha256:" + hashlib.sha256(
        _blob_path(layout, manifest_digest).read_bytes()).hexdigest()
    checks.append(VerifyCheck(
        "manifest integrity", actual_manifest == manifest_digest,
        "" if actual_manifest == manifest_digest else f"got {actual_manifest}"))

    # 2. blob self-consistency (config + each layer)
    for desc in [manifest.get("config") or {}] + (manifest.get("layers") or []):
        digest = desc.get("digest", "")
        media = desc.get("mediaType", "?")
        if not digest:
            continue
        blob = _blob_path(layout, digest)
        if not blob.is_file():
            checks.append(VerifyCheck(f"blob present: {media}", False, "missing on disk"))
            continue
        got = "sha256:" + hashlib.sha256(blob.read_bytes()).hexdigest()
        checks.append(VerifyCheck(
            f"blob self-consistency: {media}", got == digest,
            "" if got == digest else f"got {got}"))

    library_digest = next(
        (layer.get("digest") for layer in (manifest.get("layers") or [])
         if layer.get("mediaType") == PUX_LIBRARY_LAYER_MEDIATYPE), None)

    # 3. manifest trust anchor
    if expected:
        checks.append(VerifyCheck(
            "manifest == expected trust anchor", manifest_digest == expected,
            "" if manifest_digest == expected else f"expected {expected}"))
    # 4. library trust anchor
    if expected_library and library_digest:
        checks.append(VerifyCheck(
            "library == expected trust anchor", library_digest == expected_library,
            "" if library_digest == expected_library else f"expected {expected_library}"))
    # 5. source attestation
    if org:
        derived = _library_layer_digest_from_source(org, source_root)
        if derived is not None and library_digest:
            checks.append(VerifyCheck(
                f"library matches source orgs/{org}/lib/", derived == library_digest,
                "" if derived == library_digest else f"source-derived {derived}"))

    return VerifyResult(
        ok=all(chk.ok for chk in checks), layout=layout, manifest_digest=manifest_digest,
        library_digest=library_digest, checks=checks)
