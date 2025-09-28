from dotenv import load_dotenv
import os  
from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor
from tools import search_tool,wiki_tool,save_tool 

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm3 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)
# llm = ChatOpenAI(model="gpt-4o-mini")
# llm2 = ChatAnthropic(model="claude-2")


# response=llm3.invoke("What is life?")
# print(response)

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
parser=PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are world's top researcher (Top 1 percent in your field) who generates academic papers.
            Always use deep research, advanced organization, and clear formatting.
            Answer all queries using LLM tools and cite each factual statement.
            Wrap your output ONLY in this format:\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        (
            "human",
            "{query}"
        ),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools=[search_tool,wiki_tool,save_tool]
agent=create_tool_calling_agent(
llm=llm3,
prompt=prompt,
tools=tools,
)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
query=input("What do you want to research? ")
raw_response=agent_executor.invoke({"query":query})

# raw_response=agent_executor.invoke({"query":"Write a research paper on the topic 'AI in Healthcare' with relevant sources and tools used."})
print(raw_response)

try:
    structured_response=parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print(f"Error parsing response: {e}","Raw response:",raw_response)

