from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

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
                "You are very powerful assistant, but don't know current events. To calculate with emt use the calculator string emt and to calculate with mace use the calculator string mace. For each query vailidate that it contains a chemical element and a calculator string and otherwise use Alumninum as the default chemical element and emt as the default calculator string.",
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