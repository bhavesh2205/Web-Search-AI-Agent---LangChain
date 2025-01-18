### Web Search Multi AI Agent 🌐

This project is a web application that integrates multiple AI tools, enabling a multi-agent environment to perform web searches, Wikipedia queries, and leverage Groq models for natural language processing tasks. The app uses LangChain to orchestrate a ReAct agent that can process user input and provide responses based on various data sources, including DuckDuckGo and Wikipedia. By using multiple tools, the app aims to offer a comprehensive and interactive AI experience.

The core functionality of this application is built on LangChain, a framework that allows you to easily integrate multiple AI models and tools. It leverages **DuckDuckGoSearchResults for web searches**, the **WikipediaAPIWrapper for accessing detailed information from Wikipedia**, and a ChatGroq model for natural language understanding and response generation. The agents can think, act, and provide rich answers based on real-time data, ensuring users have up-to-date and reliable information.

For deployment, this app is set up on **Render**, making it easily accessible on the web. It allows seamless user interaction via the Streamlit interface, where users can ask questions related to any topic. The app intelligently uses its tools to provide relevant information, either through search results or Wikipedia summaries. The use of Groq’s language models ensures that the app offers sophisticated answers to even complex queries.