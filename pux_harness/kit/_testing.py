"""Shared offline test/verify helpers for the kit — a scripted ``BaseChatModel``.

Lives in ``kit/`` (the slim core) so ``export.verify_export`` and every test
suite import ONE canonical scripted model instead of each re-declaring its own.
Imports ONLY ``langchain_core`` — the kit import-isolation AST tripwire
(``tests/kit/test_kit_import_isolation.py``) and the runtime isolation check
(``test_kit_compile.py::test_import_isolation_*``) both hold, and
``kit._testing`` is added to that runtime check so this file's slimness is a
permanent contract.

This is a PRIVATE module (``_`` prefix): NOT re-exported from ``kit/__init__``,
so it never joins the public surface. It exists so the export verify gate
(``pux_harness.export.verify_export``) and the round-trip-compile contract test
cannot drift apart — they share the exact same compile-only model. Tests that
need to actually DRIVE a tool call subclass ``ScriptedModel`` and override
``_generate`` (see ``tests/kit/test_kit_compile.py``).
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(BaseChatModel):
    """Minimal compile-only ``BaseChatModel``.

    ``compile_org`` builds the graph structure WITHOUT ever invoking the model,
    so ``_generate`` never runs — the instance only has to satisfy the
    ``BaseChatModel`` type so the graph wires. ``bind_tools`` returns ``self``
    (the proven no-op used by every scripted-model test in this repo).
    """

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )


def scripted_model() -> ScriptedModel:
    """Factory — a fresh compile-only ``ScriptedModel`` (stateless)."""
    return ScriptedModel()
