from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


SYSTEM_PROMPT = """
You are very powerful assistant, but don't know current events.
You can calculate materials properties with EMT and MACE.
- Effective medium theory (EMT) is a cheap, analytical model that describes the macroscopic properties of composite materials, but is not available for all elements. 
  To calculate with EMT use the calculator string "emt".
- MACE is a powerful machine learning force field for predicting many-body atomic interactions that covers the periodic table.
  To calculate with MACE use the calculator string "mace".
If the user does not specify a calculator string, ask the user to provide one.

If no chemical element is provided, use Aluminum as the default chemical element.
"""
# and emt as the default calculator string

def get_executor(OPENAI_API_KEY):
    from llm_compmat.tools import interface

    llm = ChatOpenAI(
        model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY
        #model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY
    )
    tools = [
        interface.get_equilibrium_volume,
        interface.get_equilibrium_lattice,
        interface.plot_equation_of_state,
        interface.get_bulk_modulus,
        interface.get_experimental_elastic_property_wikipedia,
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            ("placeholder", "{conversation}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm_with_tools = llm.bind_tools(tools)
    agent = (
        {
            "conversation": lambda x: x["conversation"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    return AgentExecutor(agent=agent, tools=tools, verbose=True)