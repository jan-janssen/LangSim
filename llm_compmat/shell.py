from llm_compmat.llm import get_executor
import matplotlib.pyplot as plt


def dialog(OPENAI_API_KEY):
    agent_executor = get_executor(OPENAI_API_KEY=OPENAI_API_KEY)
    while True:
        lst = list(agent_executor.stream({"conversation": [("user", input())]}))
        plt.show()
