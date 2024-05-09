from llm_compmat.llm import get_extension
import matplotlib.pyplot as plt


def dialog(openapi_token=None, groq_token=None, temperature=0, **kwargs):
    agent_executor = get_extension(openapi_token=openapi_token, groq_token=groq_token, temperature=temperature, **kwargs)
    while True:
        lst = list(agent_executor.stream({"input": input()}))
        plt.show()
