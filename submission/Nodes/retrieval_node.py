from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from submission.Utils.agent_state import AgentState


class RetrievalNode():
    """Retrieves core information from dialog of user and assistant."""

    def __init__(self, llm: BaseChatModel, prompt: str, log=False) -> None:
        self.extraction_model = PromptTemplate.from_template(prompt) | llm | StrOutputParser()
        self.log = log

    def invoke(self, state: AgentState) -> AgentState:
        history = "\n".join([f"{msg.role}: {msg.content}" for msg in state["messages"][1:]
                             if isinstance(msg, (HumanMessage, AIMessage))])
        response = self.extraction_model.invoke({"history": history})

        if self.log is True:
            print("RetrievalNode")
            print(f"MSGS: {state['messages']}")
            print(f"Response: {response}")
            print()

        message = FunctionMessage(
            content=str(response),
            name="retrieve_func",
        )
        messages = state["messages"] + [message]

        return {"messages": messages, "requirements": state["requirements"]}
