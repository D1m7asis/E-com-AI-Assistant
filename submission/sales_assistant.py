from typing import List

from langchain_community.chat_models.gigachat import GigaChat
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph.graph import CompiledGraph

from submission.Nodes.entry_node import EntryNode
from submission.Nodes.checker_node import CheckerNode
from submission.Utils.prompts import CHECKER_PROMPT_TEMPLATE, RETRIEVE_PROMPT_TEMPLATE, CATEGORY_NODE_TEMPLATE, PRICE_NODE_TEMPLATE, \
    FEATURES_EXTRACTOR_TEMPLATE, RECOMMENDER_NODE_TEMPLATE, DECIDER_PROMPT_TEMPLATE, FEATURES_NODE_TEMPLATE
from submission.Nodes.retrieval_node import RetrievalNode
from submission.Nodes.category_node import CategoryNode
from submission.Nodes.price_node import PriceNode
from submission.Nodes.features_node import FeaturesNode
from submission.Nodes.recommender_node import RecommenderNode
from submission.Utils.routes import RelevantInformationRoute, TypeOfInformationRoute
from langgraph.graph import StateGraph, END
import polars as pl

from submission.Utils.agent_state import AgentState


class SalesAssistant:
    """Baseline assistant class to perform recommendations based on user preferences."""

    def __init__(
            self,
            gigachat_credentials: str,
            gigachat_scope: str,
            categories: pl.DataFrame,
            items: pl.DataFrame
    ) -> None:
        self.gigachat_credentials = gigachat_credentials
        self.gigachat_scope = gigachat_scope
        self.categories = categories
        self.items = items
        self.graph = self._build_graph(log=False)
        self.messages = []
        self.requirements = {}

    def _build_graph(self, log=False) -> CompiledGraph:
        llm = GigaChat(
            verify_ssl_certs=False,
            timeout=6000,
            temperature=0.00000000001,
            model="GigaChat-Pro",  # FIXME на деплой лучше ставить GigaChat-Pro
            credentials=self.gigachat_credentials,
            scope=self.gigachat_scope,
            retries=3,
            verbose=log
        )

        entry_node = EntryNode()
        checker_node = CheckerNode(llm, CHECKER_PROMPT_TEMPLATE, log=log)
        retrieval_node = RetrievalNode(llm, RETRIEVE_PROMPT_TEMPLATE, log=log)
        category_node = CategoryNode(llm, CATEGORY_NODE_TEMPLATE, self.categories, log=log)
        price_node = PriceNode(llm, PRICE_NODE_TEMPLATE, log=log)
        features_node = FeaturesNode(llm, FEATURES_EXTRACTOR_TEMPLATE, FEATURES_NODE_TEMPLATE, self.categories, log=log)
        recommender_node = RecommenderNode(llm, RECOMMENDER_NODE_TEMPLATE, self.items, log=log)
        decider_route = RelevantInformationRoute(llm, DECIDER_PROMPT_TEMPLATE, log=log)
        information_route = TypeOfInformationRoute(log=log)

        workflow = StateGraph(AgentState)
        workflow.add_node("entry", entry_node.invoke)
        workflow.add_node("checker", checker_node.invoke)
        workflow.add_node("retrieval", retrieval_node.invoke)
        workflow.add_node("category", category_node.invoke)
        workflow.add_node("price", price_node.invoke)
        workflow.add_node("features", features_node.invoke)
        workflow.add_node("recommender", recommender_node.invoke)
        workflow.add_conditional_edges(
            "entry",
            decider_route.invoke,
            {
                "checker": "checker",
                "retrieval": "retrieval",
            },
        )
        workflow.add_edge("checker", END)
        workflow.add_conditional_edges(
            "retrieval",
            information_route.invoke,
            {
                "category_node": "category",
                "price_node": "price",
                "features_node": "features",
                "recommender_node": "recommender",
                "end": END,
            },
        )
        workflow.add_conditional_edges(
            "category",
            category_node.route,
            {
                "price_node": "price",
                "end": END,
            }
        )
        workflow.add_conditional_edges(
            "price",
            price_node.route,
            {
                "features_node": "features",
                "end": END,
            }
        )
        workflow.add_conditional_edges(
            "features",
            features_node.route,
            {
                "recommender_node": "recommender",
                "end": END,
            }
        )
        workflow.add_edge("recommender", END)
        workflow.set_entry_point("entry")
        graph = workflow.compile()

        return graph

    def start(self) -> str:
        start_message = "Привет, я AI консультант для помощи в выборе товаров в интернет магазине. Что вы ищете?"
        self.messages.append(
            AIMessage(
                content=start_message,
                role="assistant",
            )
        )
        return start_message

    def chat(self, message: str) -> str:
        self.messages.append(HumanMessage(content=message, role="user"))
        inputs = {"messages": self.get_history(), "requirements": self.requirements}
        outputs = self.graph.invoke(inputs)

        self.messages = outputs["messages"]
        return self.messages[-1].content

    def get_history(self) -> List[BaseMessage]:
        return self.messages
