"""Upstream-protocol runtime bindings.

This package is the SEAM where pux hands a protocol surface to an UPSTREAM
runtime instead of serving a hand-rolled one:

* :mod:`pux_harness.runtime.upstream` — declares pux's graphs for ``langgraph-api``
  (the official Agent Protocol server) via ``langgraph.json``. Replaces the
  hand-rolled REST lane that was ``server.py`` (RETIRED in Aegra phase D; the
  pi-pivot: downstream → upstream). One AP runtime owner: Aegra.
"""
