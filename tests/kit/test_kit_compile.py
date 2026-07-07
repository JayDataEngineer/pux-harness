"""End-to-end compile tests for the kit: build a temp org, compile it via
``compile_org`` against a scripted (offline) model, and invoke the resulting
graph — proving the full path works with NO Docker/context/memory deps.

The scripted-model pattern mirrors ``tests/test_context_subagent.py`` (the
harness's proven fake): a ``BaseChatModel`` whose ``bind_tools`` returns self and
whose ``_generate`` emits one canned tool call, then a plain terminator.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, PrivateAttr

from pux_harness.kit import compile_org, load_subagents
from pux_harness.kit._testing import ScriptedModel


# --- a scripted model + a stub tool (offline; no API key) -------------------

class _NoArgs(BaseModel):
    pass


def _stub_tool() -> StructuredTool:
    def _run(prompt: str = "x") -> str:
        return f"FORM({prompt})"

    class _Args(BaseModel):
        prompt: str = Field("x")

    return StructuredTool.from_function(
        _run, name="generate_form", args_schema=_Args,
        description="Emit a form.",
    )


class _ScriptedModel(ScriptedModel):
    """Call ``generate_form`` once, then end the turn.

    Inherits ``_llm_type`` + ``bind_tools`` from the shared compile-only
    ``ScriptedModel`` (``kit/_testing.py``); only ``_generate`` is overridden
    here to drive a real tool call for the invoke-path tests below.
    """

    _calls: int = PrivateAttr(default=0)

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self._calls += 1
        if self._calls == 1:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "generate_form", "args": {"prompt": "koi"}, "id": "c1"}],
            )
        else:
            msg = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def solo_tree(tmp_path: Path) -> Path:
    """An org with an EMPTY roster — just a supervisor + the consumer's tool.
    Proves the simplest case: compile + invoke reach the stub tool."""
    (tmp_path / "AGENTS.md").write_text("# Base\n\nAssistant.\n")
    org = tmp_path / "orgs" / "solo"
    org.mkdir(parents=True)
    (org / "AGENTS.md").write_text("# Solo\n\nOverlay.\n")
    (org / "org.yaml").write_text("agents: []\n")
    return tmp_path


@pytest.fixture
def roster_tree(tmp_path: Path) -> Path:
    """An org that delegates to a ``form_builder`` specialist carrying a skill.
    Proves subagents compile and the skill resolves to a LOCAL path."""
    (tmp_path / "AGENTS.md").write_text("# Base\n\nAssistant.\n")
    org = tmp_path / "orgs" / "wan2"
    (org / "agents").mkdir(parents=True)
    (org / "skills" / "wan2gp").mkdir(parents=True)
    (org / "AGENTS.md").write_text("# Wan2\n\nOverlay.\n")
    (org / "org.yaml").write_text("agents: [form_builder]\n")
    (org / "agents" / "form_builder.md").write_text(
        "---\nname: form_builder\ndescription: d\ntools: [generate_form]\n"
        "skills: [orgs/wan2/skills]\n---\n\n# Form Builder\n\nBuild forms.\n"
    )
    (org / "skills" / "wan2gp" / "SKILL.md").write_text(
        "---\nname: wan2gp\ndescription: d\n---\n\n# Wan2GP\n\nparams.\n"
    )
    return tmp_path


# --- tests ------------------------------------------------------------------

def test_compile_org_invokes_and_reaches_tool(solo_tree: Path) -> None:
    """The load-bearing E2E: compile an org and invoke it; the supervisor reaches
    the consumer's stub tool — no Docker, no context layer, no model registry."""
    tool = _stub_tool()
    graph = compile_org(
        "solo",
        model=_ScriptedModel(),
        tools=[tool],
        project_root=solo_tree,
        checkpointer=__import__("langgraph.checkpoint.memory", fromlist=["MemorySaver"]).MemorySaver(),
    )
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "build the form"}]},
        config={"configurable": {"thread_id": "t1"}},
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected generate_form to be called"
    assert tool_msgs[-1].content == "FORM(koi)"


def test_load_subagents_resolves_skill_locally(roster_tree: Path) -> None:
    """A specialist with a ``skills:`` frontmatter resolves to an ABSOLUTE local
    path under project_root (kit mode), NOT a /sandbox/workspace container path."""
    tool = _stub_tool()
    subs = load_subagents("wan2", [tool], project_root=roster_tree, model="fake-model")
    assert len(subs) == 1
    sub = subs[0]
    assert sub["name"] == "form_builder"
    assert sub["tools"] == [tool]            # exact-name whitelist hit
    assert sub["model"] == "fake-model"      # default resolver → supervisor model
    assert sub["skills"] == [str(roster_tree / "orgs" / "wan2" / "skills")]


def test_load_subagents_skips_unknown_tools(roster_tree: Path) -> None:
    """An org authored under the pux harness may reference tools a standalone
    consumer doesn't ship (e.g. pux_sandbox_*). The kit SKIPS them, not raises."""
    (roster_tree / "orgs" / "wan2" / "agents" / "form_builder.md").write_text(
        "---\nname: form_builder\ndescription: d\n"
        "tools: [generate_form, pux_sandbox_execute, shared/researcher]\n"
        "---\n\n# Form Builder\n\nBuild forms.\n"
    )
    tool = _stub_tool()
    subs = load_subagents("wan2", [tool], project_root=roster_tree, model="fake-model")
    assert subs[0]["tools"] == [tool]  # only generate_form; the rest skipped silently


def test_compile_org_with_subagent_and_skills(roster_tree: Path) -> None:
    """Full compile of an org that delegates to a specialist carrying a skill.
    Builds a graph (subagent compiled in) without invoking."""
    tool = _stub_tool()
    graph = compile_org(
        "wan2",
        model=_ScriptedModel(),
        tools=[tool],
        project_root=roster_tree,
        checkpointer=None,
    )
    # the graph is a CompiledStateGraph; invoking its nodes should not raise
    # during compile. We assert it has a graph structure (nodes/edges compiled).
    assert hasattr(graph, "nodes")
    assert hasattr(graph, "stream")


def test_import_isolation_no_docker_no_heavy_subsystem() -> None:
    """LOAD-BEARING slim-core contract: importing the kit pulls in NEITHER
    ``docker`` NOR any heavy ``pux_harness`` subsystem (sandbox/browser/context).
    The kit IS ``pux_harness.kit`` now (folded in-tree, Stage 1), so the parent
    ``pux_harness`` package is expected in ``sys.modules`` — what must stay OUT
    is the Docker sandbox + the heavy optional subsystems (the lazy-import +
    tripwire firewall that replaces the old package boundary). Run in a fresh
    subprocess so prior test imports don't pollute ``sys.modules``."""
    code = (
        "import sys; import pux_harness.kit; import pux_harness.kit.compile; "
        "import pux_harness.kit.loaders; import pux_harness.kit._testing; "
        "mods = sys.modules; "
        "heavy = ['docker', 'pux_harness.sandbox', 'pux_harness.browser', "
        "'pux_harness.context']; "
        "print(repr([m for m in heavy if m in mods]))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    leaked = eval(out)
    assert leaked == [], f"kit pulled heavy deps: {leaked}"
