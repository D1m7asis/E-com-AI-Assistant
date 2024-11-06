from submission.Utils.agent_state import AgentState


class EntryNode:
    """Entry node of graph."""

    def __init__(self) -> None:
        pass

    def invoke(self, state: AgentState) -> AgentState:
        return {"messages": state["messages"], "requirements": state["requirements"]}
