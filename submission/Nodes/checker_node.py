from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from .base_node import BaseNode
from submission.Utils.agent_state import AgentState
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser


class CheckerNode(BaseNode):
    """
    Gets the answer of language model based on graph state and dialog between user and assistant
    and describes why user input is off-topic.
    """

    def __init__(self, llm: BaseChatModel, prompt: str, log=False) -> None:
        self.chain = PromptTemplate.from_template(prompt) | llm | StrOutputParser()
        self.log = log

    def invoke(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        response = self.call_model(
            last_message,
            history=state["messages"][:-1]
        )

        if self.log is True:
            print("CheckerNode")
            print(f"Message: {last_message}")
            print(f"MSGS: {state['messages']}")
            print(f"Response: {response}")
            print()

        messages = state["messages"] + [AIMessage(content=response, role="assistant")]

        return {"messages": messages, "requirements": state["requirements"]}
