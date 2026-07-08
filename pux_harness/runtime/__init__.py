"""Upstream-protocol runtime bindings.

This package is the SEAM where pux hands a protocol surface to an UPSTREAM
runtime instead of serving a hand-rolled one:

* :mod:`pux_harness.runtime.upstream` — declares pux's graphs for ``langgraph-api``
  (the official Agent Protocol server) via ``langgraph.json``. Replaces the
  hand-rolled REST lane in ``server.py`` (the pi-pivot: downstream → upstream).
"""
