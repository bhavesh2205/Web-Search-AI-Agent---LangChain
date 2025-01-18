import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain.memory import ConversationBufferMemory


# load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# initialize DuckDuckGo Search
duckduckgo_search = DuckDuckGoSearchResults()

# wikipedia API Wrapper
wikipedia = WikipediaAPIWrapper()

# initialize llm model
groq_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# define tools that the agent can use
tools = [
    Tool(
        name="DuckDuckGo",
        func=duckduckgo_search.run,
        description="Use DuckDuckGo to search the web for answers."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Use Wikipedia to get detailed information about a topic."
    )
]

# pull the prompt template from the hub
prompt = hub.pull("hwchase17/react")


# add memory to the agent
memory = ConversationBufferMemory(memory_key="chat_history")

# create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=groq_model,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# streamlit App
st.title("🌐 Web Search AI Agent")
st.write("This app uses a LangChain ReAct agent with memory to answer your queries and remember the conversation.")

# display chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# input form for user query
user_input = st.text_input("Enter anything you want to search:", "")

if st.button("Search"):
    if user_input.strip():
        st.write("searching..")
        try:
            # run the agent with the user's query
            response = agent_executor.invoke({"input": user_input})
            # store the new input and response in memory
            st.session_state["chat_history"].append(f"You: {user_input}")
            st.session_state["chat_history"].append(f"Agent: {response['output']}")
            
            # display updated chat history
            st.write("#### Conversation:")
            for message in st.session_state["chat_history"]:
                st.write(message)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query before running.")