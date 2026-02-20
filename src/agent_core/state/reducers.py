"""
agent_core/state/reducers.py  —  Reusable LangGraph message-list reducers
==========================================================================
Drop-in replacements for LangGraph's built-in `add_messages` reducer that
add production-grade safeguards (context-window cap, essential-message
preservation) while remaining fully compatible with LangGraph's StateGraph.

Usage:
    from typing import Annotated
    from langchain_core.messages import BaseMessage
    from agent_core.state.reducers import prune_messages

    class MyState(TypedDict):
        messages: Annotated[list[BaseMessage], prune_messages]

Requires:
    pip install agent-core-sdk[langgraph]
    # or: pip install langgraph langchain-core
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Hard cap on the total number of messages kept in the state list.
# Adjust via subclassing or monkey-patching if a different limit is needed.
_MAX_MESSAGES = 100


def prune_messages(
    left: list[BaseMessage],
    right: list[BaseMessage],
) -> list[BaseMessage]:
    """
    LangGraph reducer that merges two message lists and enforces a hard cap
    of ``_MAX_MESSAGES`` entries while **always** preserving:

    1. **All** ``SystemMessage`` objects — the agent must never lose its
       identity, persona, or instruction set.
    2. **The first** ``HumanMessage`` — the original user task/query is the
       root context for every downstream node and must survive eviction.

    When the merged list exceeds the cap, the oldest non-essential messages
    are evicted first.  The preserved anchors + the most recent ``N`` remaining
    messages are returned, where ``N = _MAX_MESSAGES - len(preserved)``.

    Parameters
    ----------
    left : list[BaseMessage]
        The existing message list from the current LangGraph state.
    right : list[BaseMessage]
        The new messages being appended by the current node.

    Returns
    -------
    list[BaseMessage]
        The merged and (if necessary) pruned message list.

    Examples
    --------
    Normal operation (under cap) — equivalent to ``add_messages``::

        merged = prune_messages(state["messages"], [new_ai_message])

    With a full context window — oldest non-essential messages are evicted::

        # 120 messages → trimmed to 100, always keeping SystemMessage + first HumanMessage
        pruned = prune_messages(existing_120, [next_message])
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    merged = add_messages(left, right)
    if len(merged) <= _MAX_MESSAGES:
        return merged

    # ── Phase 1: collect essential anchors ──────────────────────────────────
    preserved: list[BaseMessage] = []

    # Keep ALL SystemMessages (agent identity / instructions)
    for msg in merged:
        if isinstance(msg, SystemMessage):
            preserved.append(msg)

    # Keep the very first HumanMessage (original task query)
    for msg in merged:
        if isinstance(msg, HumanMessage) and msg not in preserved:
            preserved.append(msg)
            break

    # ── Phase 2: fill remaining slots with recency ───────────────────────────
    remaining_slots = _MAX_MESSAGES - len(preserved)
    if remaining_slots <= 0:
        # Edge case: essential messages alone exceed the cap (extremely rare).
        # Return only the preserved anchors rather than violating the cap.
        return preserved

    others = [m for m in merged if m not in preserved]
    recent = others[-remaining_slots:]

    return preserved + recent
