from getpass import getpass
from llm_compmat.llm import get_extension


def dialog():
    agent_executor = get_extension(OPENAI_API_KEY=getpass(prompt='Enter your OpenAI Token:'))
    while True:
        lst = list(agent_executor.stream({"input": input()}))
        print(lst)