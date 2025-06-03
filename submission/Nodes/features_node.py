from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from submission.Utils.agent_state import AgentState
from .base_node import BaseNode
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
import polars as pl
import json


class FeaturesNode(BaseNode):
    """Extracts the additional features of item based on user answers."""

    def __init__(
            self,
            llm: BaseChatModel,
            extractor_prompt: str,
            prompt: str,
            categories_dataframe: pl.DataFrame,
            log=False,
    ) -> None:
        self.questions = []
        self.extractor = (
            PromptTemplate.from_template(extractor_prompt)
            | llm
            | StrOutputParser()
        )
        self.chain = (
            PromptTemplate.from_template(prompt)
            | llm
            | StrOutputParser()
        )
        self.categories_dict = {row["id"]: row["name"] for row in categories_dataframe.to_dicts()}
        self.log = log
        self.first = True

    def invoke(self, state: AgentState) -> AgentState:
        last_message = state["messages"][-1]

        response = {}

        if isinstance(last_message, FunctionMessage):
            # Суммарное сообщение от RetrievalNode содержит все предпочтения
            try:
                extractor_invoke = self.extractor.invoke({"query": last_message.content})
                response = json.loads(extractor_invoke)
            except Exception as e:
                if self.log:
                    print("Exception while parsing direct requirements in features node: " + str(e))

            if "цена" in response:
                del response["цена"]

            state["requirements"].update(response)

            if self.first or not self.questions:
                category_name = self.categories_dict.get(
                    state["requirements"].get("category"), "Unknown Category"
                )
                llm_questions = self.chain.invoke({"messages": last_message.content, "catalog": category_name}).replace("'", '"')
                try:
                    self.questions = json.loads(llm_questions)
                except Exception as e:
                    if self.log:
                        print("Exception while parsing llm_questions in features node: " + str(e))
                self.first = False
        else:
            dialog = self.get_last_dialog_messages(state["messages"], count=2)
            try:
                response = json.loads(self.extractor.invoke({"query": dialog}))
            except Exception as e:
                if self.log:
                    print("Exception while parsing extractor in features node: " + str(e))

            state["requirements"].update(response)

        if self.questions:
            state["messages"].append(AIMessage(content=self.questions.pop(0), role="assistant"))

        if self.log:
            print("FeaturesNode")
            print()

        return state

    def route(self, state: AgentState) -> str:
        if isinstance(state["messages"][-1], AIMessage):
            return "end"
        else:
            return "recommender_node"

    def get_last_dialog_messages(self, messages, count=2):
        """Извлекаем последние сообщения от человека и AI."""
        return "\n".join([f"{msg.role}: {msg.content}" for msg in messages if isinstance(msg, (HumanMessage, AIMessage))][-count:])
