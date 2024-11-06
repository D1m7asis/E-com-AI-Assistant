from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    requirements: Annotated[dict, operator.setitem]
    messages: Annotated[Sequence[BaseMessage], operator.setitem]
