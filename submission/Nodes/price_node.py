from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from submission.Utils.agent_state import AgentState
from .base_node import BaseNode
from langchain_core.messages import AIMessage, FunctionMessage


class PriceNode(BaseNode):
    """Extracts the budget of user."""

    def __init__(self, llm: BaseChatModel, prompt: str, log=False) -> None:
        self.chain = PromptTemplate.from_template(prompt) | llm | StrOutputParser()
        self.log = log

    def invoke(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        requirements = state["requirements"]

        last_message = messages[-1]
        assert isinstance(last_message, FunctionMessage)

        response = self.call_model(last_message.content)

        try:
            price = int(response)
            requirements["price"] = price
        except ValueError:
            messages += [AIMessage(content=response, role="assistant")]

        if self.log is True:
            print("PriceNode")
            print(f"Query: {last_message.content}")
            print(f"MSGS: {state['messages']}")
            print(f"Response: {response}")
            print()

        return {"messages": messages, "requirements": requirements}

    def route(self, state: AgentState) -> str:
        if isinstance(state["messages"][-1], AIMessage):
            return "end"
        else:
            return "features_node"
