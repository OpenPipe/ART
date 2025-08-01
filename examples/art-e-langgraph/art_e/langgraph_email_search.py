"""
LangGraph Email Search Agent using create_react_agent

This module implements an email search agent using LangGraph's create_react_agent
that can search and read emails using the existing email search tools.
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from art.langgraph import init_chat_model
import uuid

load_dotenv()


logger = logging.getLogger(__name__)


class EmailSearchReActAgent:
    """ReAct-style email search agent using LangGraph"""

    def __init__(self, model=None, tools=None):
        # Use ChatOpenAI for real LLM
        if model is None:
            model = init_chat_model("gpt-4o-mini")

        # Create the ReAct agent
        self.agent = create_react_agent(model, tools)

    async def search(self, system_prompt, human_prompt) -> str:
        try:
            # Run the agent
            config = {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": 20,
            }
            await self.agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt),
                    ]
                },
                config=config,
            )
        except Exception as e:
            logger.error(f"Error running ReAct agent: {e}")
            return f"Error processing query: {str(e)}"


def main():
    """Demo the ReAct email search agent"""

    print("LangGraph ReAct Email Search Agent Demo")
    print("=" * 50)

    # Initialize agent
    react_agent = EmailSearchReActAgent()

    # Example queries
    queries = [
        "Find emails about California energy crisis",
        "Show emails from jeff.dasovich@enron.com about power",
        "Search for emails mentioning FERC",
    ]

    print("\nReAct Email Search Agent:")
    print("-" * 30)

    for query in queries:
        print(f"\nQuery: {query}")
        print("Result:")
        result = react_agent.search(query)
        print(result)
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
