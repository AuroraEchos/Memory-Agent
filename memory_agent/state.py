"""LangGraph state schema for the Memory Agent workflow."""

from dataclasses import dataclass, field
from typing import Any
from typing_extensions import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class State:
    """Conversation state carried between LangGraph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
