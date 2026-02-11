import os

from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)

agent = create_agent(
    model=model,
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked to create an optimized query (less than 50 words) using the provided user query."
            }
        ]
    )    
)

def main():
    response = agent.invoke(
        {
            "messages": [HumanMessage("What county program is for fiefigters in my county?")]
        }
    )
    
    print(response)
    
if __name__ == "__main__":
    main()