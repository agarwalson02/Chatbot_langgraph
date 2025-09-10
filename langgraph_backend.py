from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage 
from langchain_nebius import ChatNebius
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

llm=ChatNebius(model="openai/gpt-oss-120b")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState):
    messages=state['messages'] #take previous messages and query from messages
    response=llm.invoke(messages) #send to llm
    return {'messages':[response]} #response added to messages 

checkpointer=InMemorySaver()

graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)

