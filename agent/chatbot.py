import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, MessagesState
from menu_handler import handle_menu   # ✅ Import menu handler

load_dotenv()

# Initialize Groq LLM with supported model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"   # ✅ supported model
)

# Define how chatbot responds
def chatbot_node(state: MessagesState):
    last_message = state["messages"][-1].content
    menu_reply = handle_menu(last_message, state["messages"])

    if menu_reply:  # ✅ intercept menu flow (guided options)
        return {"messages": state["messages"] + [menu_reply]}
    
    # Fallback → use Groq LLM for free chat
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# Build LangGraph workflow
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
chatbot_app = graph.compile()

if __name__ == "__main__":
    print("🍳 Kitchen Chatbot is ready! Type 'exit' to quit.\n")
    conversation = {"messages": []}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        conversation["messages"].append(HumanMessage(content=user_input))
        conversation = chatbot_app.invoke(conversation)   # ✅ invoke LangGraph
        bot_reply = conversation["messages"][-1].content
        print(f"Bot: {bot_reply}\n")
