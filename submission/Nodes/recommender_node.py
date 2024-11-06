import json
from pathlib import Path

import chromadb
import polars as pl
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from submission.Utils.agent_state import AgentState
from .base_node import BaseNode
from langchain_core.messages import AIMessage, FunctionMessage

from ..Utils.prompts import LLM_PRETTIFY_PROMPT_TEMPLATE
from ..Utils.utils import get_chroma_collection, search_items

chroma_client = chromadb.PersistentClient(
    path=str(Path(__file__).parent.resolve() / "items_chroma")
)
items_collection = get_chroma_collection(chroma_client, "items")


class RecommenderNode(BaseNode):
    """Recommends 10 items based on category, budget and additional features."""

    def __init__(
        self, llm: BaseChatModel, prompt: str, items_dataframe: pl.DataFrame, log=False
    ) -> None:
        self.chain = PromptTemplate.from_template(prompt) | llm | StrOutputParser()
        self.llm_prettifier = (
            PromptTemplate.from_template(LLM_PRETTIFY_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        )
        self.items_dataframe = items_dataframe
        self.log = log

    def invoke(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        assert isinstance(last_message, FunctionMessage)

        vector_top = search_items(self.llm_prettifier.invoke({"query": last_message, "requirements": state["requirements"]}),
                                  items_collection,
                                  20,
                                  state["requirements"]["category"],)

        response = self.retrieve_items(state["requirements"], context=vector_top)

        if self.log is True:
            print("RecommenderNode")
            print()

        messages = state["messages"] + [AIMessage(content=response, role="assistant")]

        return {"messages": messages, "requirements": state["requirements"]}

    def retrieve_items(self, query, context) -> str:
        return self.call_model(str(query), context=context)
