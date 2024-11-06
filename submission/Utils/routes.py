from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .agent_state import AgentState
from submission.Nodes.base_node import BaseNode


class RelevantInformationRoute(BaseNode):
    """Route that decides is user query input is relevant or not."""

    IDX_TO_NODE = {
        0: "checker",
        1: "retrieval",
    }

    def __init__(self, llm: BaseChatModel, prompt: str, log=False) -> None:
        self.chain = PromptTemplate.from_template(prompt) | llm | StrOutputParser()
        self.log = log

    def invoke(self, state: AgentState) -> str:
        """
        Returns mapping of which node to call next:
        0 - if user input has logical issues or contains off-topic information.
        1 - otherwise.

        Args:
            state (messages): Current graph state.

        Returns:
            str: Next node name.
        """
        last_message = state["messages"][-1]
        decision = int(self.call_model(last_message, history=state["messages"][:-1]))

        if self.log is True:
            print("DeciderRoute")
            print(f"Message: {last_message}")
            print(f"Decision: {RelevantInformationRoute.IDX_TO_NODE[decision]}")
            print()

        return RelevantInformationRoute.IDX_TO_NODE[decision]


class TypeOfInformationRoute:
    """Route decides which information is not enough for recommendations and calls appropriate node."""

    def __init__(self, log=False) -> None:
        self.log = log
        self.is_recommended = False

    def invoke(self, state: AgentState) -> str:
        data = state["requirements"]
        if self.log is True:
            print("TypeOfInformationRoute")
            print(data)
            print()

        if "category" not in data:
            return "category_node"
        elif "price" not in data:
            return "price_node"
        elif len(data) <= 5:
            return "features_node"
        elif not self.is_recommended:
            self.is_recommended = True
            return "recommender_node"
        else:
            return "end"
