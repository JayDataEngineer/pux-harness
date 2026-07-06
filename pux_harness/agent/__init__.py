"""Agent assembly layer: deepagents graph builder + per-org system prompts,
subagent loading, the declarative org contract, and the model factory.

Consumed by the top-level entry points (``main``/``server``/``cli``/``acp``).
``agent`` depends on ``sandbox`` (specialist tools, the model is wired into
``describe_image``) and ``context`` (the offload middleware rides the graph)."""
