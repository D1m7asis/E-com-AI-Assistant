from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage

from submission.Utils.agent_state import AgentState
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel


class BaseNode(ABC):
    """Base node of graph."""

    def __init__(self, llm: BaseChatModel, prompt: PromptTemplate) -> None:
        self.llm = llm
        pass

    @abstractmethod
    def invoke(self, state: AgentState) -> AgentState:
        pass

    def call_model(self, query: str, **kwargs) -> str:
        """
        Calls LLMChain with additional args.

        Args:
            query (str): User query.
            **kwargs (dict): additional arguments to pass into template.

        Returns:
            str: Result of execution of LLMChain.
        """
        invoke_args = {"query": query}
        for key, value in kwargs.items():
            invoke_args[key] = value

        return self.chain.invoke(invoke_args)

    def direct_call_model(self, query: str, **kwargs) -> BaseMessage:
        return self.llm.invoke(input=query)