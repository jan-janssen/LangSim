from llm_compmat.llm import get_extension
import matplotlib.pyplot as plt


def dialog(OPENAI_API_KEY):
    agent_executor = get_extension(OPENAI_API_KEY=OPENAI_API_KEY)
    while True:
        lst = list(agent_executor.stream({"input": input()}))
        plt.show()
