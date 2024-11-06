import json
from pathlib import Path

import chromadb
import polars as pl
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from submission.Utils.agent_state import AgentState
from .base_node import BaseNode
from submission.Utils.utils import search_categories, get_chroma_collection
from langchain_core.messages import AIMessage, FunctionMessage

from ..Utils.prompts import DETERMINE_CATEGORY_TEMPLATE

chroma_client = chromadb.PersistentClient(path=str(Path(__file__).parent.resolve() / "chroma_data"))
category_collection = get_chroma_collection(chroma_client, "categories")


class CategoryNode(BaseNode):
    """Extracts the minimum subcategory of goods by questioning a user."""

    def __init__(
            self,
            llm: BaseChatModel,
            prompt: str,
            categories_dataframe: pl.DataFrame,
            log=False,
            times_to_reclassify=0
    ) -> None:
        super().__init__(llm, prompt)
        self.llm = llm
        self.relevant_categories_chain = PromptTemplate.from_template(DETERMINE_CATEGORY_TEMPLATE) | self.llm | StrOutputParser()
        self.chain = PromptTemplate.from_template(prompt) | llm | StrOutputParser()
        self.categories_dataframe = categories_dataframe
        self.log = log
        self.times_to_reclassify = times_to_reclassify

    def get_relevant_categories(self, categories_to_pick, user_query, max_retries=3):

        categories = []

        for attempt in range(max_retries):
            try:
                response = self.relevant_categories_chain.invoke({"categories_to_pick": categories_to_pick, "user_query": user_query})
                matching_categories = json.loads(response.replace("'", '"'))
                categories.extend(matching_categories)
                break
            except json.JSONDecodeError as e:
                if self.log:print(f"Попытка получения релевантных категорий {attempt + 1} не удалась: {e}")
            except Exception as e:
                if self.log: print(f"Неизвестная ошибка на попытке получения релевантных категорий {attempt + 1}: {e}")
        else:
            if self.log: print("Не удалось получить релевантные категории после всех попыток.")

        return categories

    def invoke(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        requirements = state["requirements"]

        last_message = messages[-1]
        assert isinstance(last_message, FunctionMessage)

        categories = search_categories(last_message.content, category_collection, 20)[0]
        top_categories = self.get_relevant_categories(categories, last_message.content)
        top_subcategories = self.get_subcategories(top_categories)

        if len(top_subcategories) > 0 and self.times_to_reclassify > 0:
            self.times_to_reclassify -= 1

            context = top_subcategories[:5]
            response = self.call_model(last_message.content, context=context)

            messages += [AIMessage(content=response, role="assistant")]

            if self.log is True:
                print("CategoryNode")
                print(f"Query: {last_message.content}")
                print(f"Context: {context}")
                print(f"MSGS: {state['messages']}")
                print(f"Response: {response}")
                print()
        else:
            category = category_collection.get(ids=[str(top_categories[0]["id"])])
            requirements["category"] = int(category["ids"][0])

            if self.log is True:
                print("CategoryNode")
                print(f"Query: {last_message.content}")
                print(f"requirements: {requirements}")
                print()

        return {"messages": messages, "requirements": requirements}

    def get_subcategories(self, categories):
        if categories:
            return self.categories_dataframe.filter(parent_id=categories[0]["id"]).to_dict()["name"].to_list()

    def route(self, state: AgentState) -> str:
        if isinstance(state["messages"][-1], AIMessage):
            return "end"
        else:
            return "price_node"
