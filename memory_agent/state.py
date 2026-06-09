from dataclasses import dataclass
from typing_extensions import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class State:
    messages: Annotated[list[AnyMessage], add_messages]