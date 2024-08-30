from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description_and_args
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)

from langsim.tools.interface import (
    get_bulk_modulus,
    get_equilibrium_volume,
    get_experimental_elastic_property_wikipedia,
    plot_equation_of_state,
    get_atom_dict_bulk_structure,
    get_atom_dict_equilibrated_structure,
)
from langsim.prompt import SYSTEM_PROMPT_ALT, SYSTEM_PROMPT
from langchain import hub


def get_executor(
    api_provider,
    api_key,
    api_url=None,
    api_model=None,
    api_temperature=0,
    agent_type="default",
):
    if api_provider.lower() == "openai":
        from langchain_openai import ChatOpenAI

        if api_model is None:
            api_model = "gpt-4o"
        llm = ChatOpenAI(
            model=api_model,
            temperature=api_temperature,
            openai_api_key=api_key,
            base_url=api_url,
        )

    elif api_provider.lower() == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if api_model is None:
            api_model = "claude-3-5-sonnet-20240620"
        llm = ChatAnthropic(
            model=api_model,
            temperature=api_temperature,
            anthropic_api_key=api_key,
        )

    elif api_provider.lower() == "groq":
        from langchain_groq import ChatGroq

        if api_model is None:
            api_model = "Llama-3-Groq-70B-Tool-Use"
        llm = ChatGroq(
            model=api_model,
            temperature=api_temperature,
            groq_api_key=api_key,
        )
    else:
        raise ValueError()

    tools = [
        get_equilibrium_volume,
        get_atom_dict_bulk_structure,
        get_atom_dict_equilibrated_structure,
        plot_equation_of_state,
        get_bulk_modulus,
        get_experimental_elastic_property_wikipedia,
    ]

    llm_with_tools = llm.bind_tools(tools)

    if agent_type == "react":
        REACT_PROMPT_FROM_HUB = hub.pull("langchain-ai/react-agent-template")
        prompt = REACT_PROMPT_FROM_HUB.partial(
            instructions=SYSTEM_PROMPT_ALT,
            tools=render_text_description_and_args(tools),
            tool_names=", ".join([t.name for t in tools]),
        )
        input_key = "input"

    elif agent_type == "default":
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{conversation}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        input_key = "conversation"

    else:
        raise ValueError()

    agent = (
        {
            input_key: lambda x: x[input_key],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
        # | JSONAgentOutputParser()
    )
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
