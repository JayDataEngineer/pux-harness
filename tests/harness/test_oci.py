"""OCI artifact emission (dynamic-tools P5) — unit mechanics via an injected oras runner.

``emit_oci_artifact`` shells out to ``oras``; here we inject a fake that mimics the
real ``oras push`` (parses the cmd, computes REAL sha256 digests over the config +
layer tars, writes a manifest.json). That makes the layer-digest / tamper-detection
contract PROVABLE offline + deterministically — the live ``oras`` round-trip is the
repo-root integration test (``tests/export/test_export_oci.py``).

Rhymes with ``test_pack_hooks.py``'s injected-runner pattern ([[verify-or-die]]).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pux_harness.oci import (
    PUX_CONFIG_MEDIATYPE,
    PUX_LIBRARY_LAYER_MEDIATYPE,
    PUX_SOURCE_LAYER_MEDIATYPE,
    OciError,
    _build_layer_tar,
    _is_library,
    _normalized_bytes,
    _split_layers,
    emit_oci_artifact,
    verify_oci_layout,
)


# --- injected oras stand-in ---------------------------------------------------

class _Proc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _digest(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


class _FakeOras:
    """Mimics ``oras push``: parses the cmd, reads the config + 2 layer tars,
    computes REAL sha256 digests, writes a manifest.json to ``--export-manifest``,
    returns JSON with the manifest digest. Offline + deterministic."""

    def __init__(self, *, available: bool = True, fail: bool = False):
        self._available = available
        self._fail = fail

    def __call__(self, cmd, **kw):
        if cmd[:2] == ["oras", "version"]:
            return _Proc(0 if self._available else 1)
        if self._fail:
            return _Proc(1, "", "simulated oras failure")
        # Walk the arg list (mirrors _run_oras_push's construction).
        config_spec = export = layout_ref = fmt = None
        layer_specs: list[str] = []
        annotations: dict[str, str] = {}
        i = 2
        while i < len(cmd):
            tok = cmd[i]
            if tok == "--oci-layout":
                layout_ref = cmd[i + 1]
                i += 2
            elif tok == "--config":
                config_spec = cmd[i + 1]
                i += 2
            elif tok == "--export-manifest":
                export = cmd[i + 1]
                i += 2
            elif tok in ("--format",):
                fmt = cmd[i + 1]  # noqa: F841
                i += 2
            elif tok == "--annotation":
                k, _, v = cmd[i + 1].partition("=")
                annotations[k] = v
                i += 2
            elif ":" in tok and not tok.startswith("--"):
                layer_specs.append(tok)
                i += 1
            else:
                i += 1

        # oras runs from a cwd; resolve the (now relative) layer/config names there
        # — mirrors the real oras invocation, which the live pull depends on.
        cwd = kw.get("cwd")

        def _resolve(name: str) -> Path:
            p = Path(name)
            if not p.is_absolute() and cwd:
                p = Path(cwd) / p
            return p

        def _desc(spec: str) -> dict:
            path, _, mt = spec.partition(":")
            b = _resolve(path).read_bytes()
            return {"mediaType": mt, "digest": _digest(b), "size": len(b),
                    "annotations": {"org.opencontainers.image.title": Path(path).name}}

        config_desc = _desc(config_spec)
        layer_descs = [_desc(s) for s in layer_specs]
        manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": config_desc, "layers": layer_descs,
            "annotations": {"org.opencontainers.image.created": "2026-07-08T00:00:00Z",
                            **annotations},
        }
        _resolve(export).write_text(json.dumps(manifest))
        manifest_digest = _digest(json.dumps(manifest, sort_keys=True).encode())
        return _Proc(0, json.dumps({
            "digest": manifest_digest,
            "reference": f"{layout_ref}@{manifest_digest}",
            "mediaType": manifest["mediaType"],
        }))


def _inputs(tmp: Path, lib_body: str = "def learned():\n    return 1\n",
             src_extra: str = "def helper():\n    return 2\n"):
    """Synthetic collected content: one lib file (the integrity target) + source
    primitives + a generated scaffold entry (the vendored kit)."""
    org_dir = tmp / "orgs" / "acme"
    (org_dir / "lib" / "functions").mkdir(parents=True, exist_ok=True)
    (org_dir / "agents").mkdir(parents=True, exist_ok=True)
    lib = org_dir / "lib" / "functions" / "learned.py"
    lib.write_text(lib_body)
    agent = org_dir / "agents" / "worker.md"
    agent.write_text(src_extra)
    files = {
        "orgs/acme/lib/functions/learned.py": lib,
        "orgs/acme/agents/worker.md": agent,
    }
    scaffold = {"run.py": b"# standalone runner\n", "pyproject.toml": b"[project]\n"}
    inventory = {"org": "acme", "total_files": 2,
                 "provenance": {"hooks": [{"name": "ast_check", "ok": True}],
                                "all_ok": True}}
    return files, scaffold, inventory


# --- layer grouping + tar -----------------------------------------------------

def test_is_library_targets_org_lib():
    assert _is_library("orgs/acme/lib/functions/learned.py", "acme")
    assert _is_library("orgs/acme/lib/index.yaml", "acme")
    assert not _is_library("orgs/acme/sandbox/helper.py", "acme")
    assert not _is_library("orgs/acme/agents/worker.md", "acme")
    assert not _is_library("run.py", "acme")


def test_is_library_does_not_misroute_a_sandbox_lib_subdir():
    """Regression: a bare ``/lib/`` substring routed ``orgs/<org>/sandbox/lib/util.py``
    into the agent-library integrity layer. Anchoring on ``/<org>/lib/`` fixes it,
    and still matches the specialists layout (``orgs/specialists/<org>/lib/``)."""
    assert not _is_library("orgs/acme/sandbox/lib/util.py", "acme")
    assert not _is_library("orgs/acme/somelib/foo.py", "acme")
    # the org segment disambiguates the specialists parent dir
    assert _is_library("orgs/specialists/invest/lib/functions/x.py", "invest")
    assert not _is_library("orgs/specialists/invest/lib/functions/x.py", "acme")


def test_split_layers_puts_lib_apart_from_source_and_scaffold(tmp_path):
    files, scaffold, _ = _inputs(tmp_path)
    source, library = _split_layers(files, scaffold, "acme")
    src_paths = [p for p, _ in source]
    lib_paths = [p for p, _ in library]
    assert "orgs/acme/lib/functions/learned.py" in lib_paths
    assert "orgs/acme/lib/functions/learned.py" not in src_paths
    # source holds the primitive + the scaffold (trusted kit, never learned lib)
    assert "orgs/acme/agents/worker.md" in src_paths
    assert "run.py" in src_paths and "pyproject.toml" in src_paths


def test_build_layer_tar_prefixes_entries_with_org():
    tar_bytes = _build_layer_tar([("orgs/acme/agents/w.md", b"x")], "acme")
    import tarfile
    from io import BytesIO
    with tarfile.open(fileobj=BytesIO(tar_bytes)) as tar:
        names = tar.getnames()
    assert names == ["acme/orgs/acme/agents/w.md"]


# --- emit ---------------------------------------------------------------------

def test_emit_raises_when_oras_absent(tmp_path):
    files, scaffold, inventory = _inputs(tmp_path)
    with pytest.raises(OciError, match="oras binary not found"):
        emit_oci_artifact("acme", files, scaffold, inventory,
                          output_layout=tmp_path / "acme.oci",
                          oras_runner=_FakeOras(available=False))


def test_emit_raises_on_oras_push_failure(tmp_path):
    files, scaffold, inventory = _inputs(tmp_path)
    with pytest.raises(OciError, match="oras push failed"):
        emit_oci_artifact("acme", files, scaffold, inventory,
                          output_layout=tmp_path / "acme.oci",
                          oras_runner=_FakeOras(fail=True))


def test_emit_produces_artifact_with_three_layers(tmp_path):
    files, scaffold, inventory = _inputs(tmp_path)
    art = emit_oci_artifact("acme", files, scaffold, inventory,
                            output_layout=tmp_path / "acme.oci",
                            oras_runner=_FakeOras())
    types = [layer.type for layer in art.layers]
    assert types == ["config", "source-code", "agent-library"]
    assert all(layer.digest.startswith("sha256:") for layer in art.layers)
    assert art.digest.startswith("sha256:")
    assert art.library_layer() is not None
    assert (tmp_path / "acme.oci" / "provenance.json").is_file()


def test_provenance_records_layers_hooks_and_reserved_signature(tmp_path):
    files, scaffold, inventory = _inputs(tmp_path)
    art = emit_oci_artifact("acme", files, scaffold, inventory,
                            output_layout=tmp_path / "acme.oci",
                            oras_runner=_FakeOras())
    prov = json.loads(art.provenance.read_text())
    assert prov["artifact"]["digest"] == art.digest
    assert prov["org"] == "acme"
    assert [layer["type"] for layer in prov["layers"]] == ["config", "source-code", "agent-library"]
    # the P4 hook results flow through into the OCI provenance audit surface
    assert prov["hooks"] == [{"name": "ast_check", "ok": True}]
    assert prov["hooks_all_ok"] is True
    assert "org.pux.org" in prov["manifest_annotations"]
    # P6 slot reserved (P5 emits unsigned; signing is later)
    assert prov["signature"] is None


def test_library_layer_digest_is_the_tamper_anchor(tmp_path):
    """Mutate a learned function → the agent-library layer digest MUST change
    (the integrity contract; the source layer digest stays put)."""
    files, scaffold, inventory = _inputs(tmp_path, lib_body="def learned():\n    return 1\n")
    clean = emit_oci_artifact("acme", files, scaffold, inventory,
                              output_layout=tmp_path / "a.oci", oras_runner=_FakeOras())
    files2, scaffold2, inventory2 = _inputs(
        tmp_path, lib_body="def EVIL():\n    return 999  # tampered\n")
    tampered = emit_oci_artifact("acme", files2, scaffold2, inventory2,
                                 output_layout=tmp_path / "b.oci", oras_runner=_FakeOras())
    assert clean.library_layer().digest != tampered.library_layer().digest, \
        "tampering the library did NOT change its digest — integrity contract broken"
    # the SOURCE layer is unaffected by a lib mutation
    clean_src = next(layer for layer in clean.layers if layer.type == "source-code")
    tamp_src = next(layer for layer in tampered.layers if layer.type == "source-code")
    assert clean_src.digest == tamp_src.digest


def test_identical_content_yields_identical_layer_digests(tmp_path):
    """Same content → same layer digests across emits (content-addressed)."""
    f1, s1, i1 = _inputs(tmp_path)
    f2, s2, i2 = _inputs(tmp_path)
    a = emit_oci_artifact("acme", f1, s1, i1, output_layout=tmp_path / "a.oci",
                          oras_runner=_FakeOras())
    b = emit_oci_artifact("acme", f2, s2, i2, output_layout=tmp_path / "b.oci",
                          oras_runner=_FakeOras())
    assert [layer.digest for layer in a.layers] == [layer.digest for layer in b.layers]


def test_media_types_are_the_pux_vocabulary(tmp_path):
    files, scaffold, inventory = _inputs(tmp_path)
    art = emit_oci_artifact("acme", files, scaffold, inventory,
                            output_layout=tmp_path / "acme.oci",
                            oras_runner=_FakeOras())
    by_type = {layer.type: layer.media_type for layer in art.layers}
    assert by_type["config"] == PUX_CONFIG_MEDIATYPE
    assert by_type["source-code"] == PUX_SOURCE_LAYER_MEDIATYPE
    assert by_type["agent-library"] == PUX_LIBRARY_LAYER_MEDIATYPE


# --- verify (close-the-loop: emit records the anchor, verify checks it) ----------
# verify reads the oci-layout DIRECTLY (stdlib-only — no oras at verify time), so the
# unit tests hand-build a REAL layout (index.json + blobs/sha256/...) on disk.

def _build_layout(
    tmp: Path, *, config: bytes = b"{}", source: bytes = b"src",
    library: bytes = b"lib", corrupt_library: bytes | None = None,
) -> tuple[Path, str, str]:
    """Hand-build a real oci-layout so verify can read it without oras. Returns
    (layout, manifest_digest, library_digest). ``corrupt_library`` writes bytes that
    do NOT hash to the recorded library digest (an in-place tamper)."""
    layout = tmp / "acme.oci"
    blobs = layout / "blobs" / "sha256"
    blobs.mkdir(parents=True)

    def _write(blob_bytes: bytes) -> str:
        digest = _digest(blob_bytes)
        (blobs / digest.split(":", 1)[1]).write_bytes(blob_bytes)
        return digest

    config_d = _write(config)
    source_d = _write(source)
    library_d = _digest(library)
    (blobs / library_d.split(":", 1)[1]).write_bytes(
        library if corrupt_library is None else corrupt_library)

    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": PUX_CONFIG_MEDIATYPE, "digest": config_d, "size": len(config)},
        "layers": [
            {"mediaType": PUX_SOURCE_LAYER_MEDIATYPE, "digest": source_d, "size": len(source)},
            {"mediaType": PUX_LIBRARY_LAYER_MEDIATYPE, "digest": library_d, "size": len(library)},
        ],
        "annotations": {"org.pux.org": "acme"},
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_d = _write(manifest_bytes)
    (layout / "index.json").write_text(json.dumps({
        "schemaVersion": 2, "manifests": [
            {"mediaType": "application/vnd.oci.image.manifest.v1+json",
             "digest": manifest_d, "size": len(manifest_bytes)}]}))
    return layout, manifest_d, library_d


def _lib_layer_tar(tmp: Path, org: str = "acme") -> bytes:
    """The agent-library layer tar built from the staged source ``orgs/<org>/lib/``
    — the SAME construction pack uses, so verify's source-attestation matches it."""
    lib_root = tmp / "orgs" / org / "lib"
    entries = []
    for path in sorted(lib_root.rglob("*")):
        if path.is_file():
            arc = path.relative_to(tmp).as_posix()
            entries.append((arc, _normalized_bytes(arc, path.read_bytes())))
    return _build_layer_tar(entries, org)


def test_verify_passes_on_a_clean_layout(tmp_path):
    layout, manifest_d, library_d = _build_layout(tmp_path)
    result = verify_oci_layout(layout)
    assert result.ok, result.summary()
    assert result.manifest_digest == manifest_d
    assert result.library_digest == library_d  # the tamper anchor is surfaced
    assert {c.name for c in result.checks if not c.ok} == set()


def test_verify_detects_a_corrupted_library_blob(tmp_path):
    """An in-place tamper of the agent-library blob (content != recorded digest) is
    caught by the blob self-consistency check — the integrity contract, on verify."""
    layout, _, _ = _build_layout(tmp_path, corrupt_library=b"TAMPERED-CONTENT")
    result = verify_oci_layout(layout)
    assert not result.ok
    failed = [c.name for c in result.checks if not c.ok]
    assert any("agent-library" in n and "self-consistency" in n for n in failed), failed


def test_verify_manifest_and_library_trust_anchors(tmp_path):
    layout, manifest_d, library_d = _build_layout(tmp_path)
    # correct anchors → pass
    assert verify_oci_layout(layout, expected=manifest_d,
                             expected_library=library_d).ok
    # wrong manifest anchor → fail on that check only
    bad_manifest = verify_oci_layout(layout, expected="sha256:" + "0" * 64)
    assert not bad_manifest.ok
    assert any("manifest == expected" in c.name for c in bad_manifest.checks if not c.ok)
    # wrong library anchor → fail
    bad_library = verify_oci_layout(layout, expected_library="sha256:" + "f" * 64)
    assert not bad_library.ok
    assert any("library == expected" in c.name for c in bad_library.checks if not c.ok)


def test_verify_source_attestation_matches_then_detects_drift(tmp_path, monkeypatch):
    """Re-deriving the agent-library layer from ``orgs/<org>/lib/`` matches the packed
    layer; mutating the learned function → the source-attestation check FAILS (the
    drift the design promises is detectable on verify)."""
    files, scaffold, inventory = _inputs(tmp_path)  # stages orgs/acme/lib/...
    library_tar = _lib_layer_tar(tmp_path)          # pack-faithful library bytes
    layout, _, _ = _build_layout(tmp_path, library=library_tar)

    matched = verify_oci_layout(layout, org="acme", source_root=tmp_path)
    assert matched.ok, matched.summary()
    assert any("matches source" in c.name for c in matched.checks)

    # the agent "learned" something new after the pack → drift
    (tmp_path / "orgs" / "acme" / "lib" / "functions" / "learned.py").write_text(
        "def learned():\n    return 999  # drifted\n")
    drifted = verify_oci_layout(layout, org="acme", source_root=tmp_path)
    assert not drifted.ok
    assert any("matches source" in c.name for c in drifted.checks if not c.ok)
