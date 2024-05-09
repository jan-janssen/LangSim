import ipywidgets as widgets
from IPython.display import display

from llm_compmat.llm import get_extension


key_widget = widgets.Text(
    value="sk-********************",
    placeholder="",
    description="OpenAI Token:",
    disabled=False,
)

query_widget = widgets.Text(
    value="Calculate the equilibrium volume of gold",
    placeholder="",
    description="Next query:",
    disabled=False,
)

out = widgets.Output()


def run_refresh(*ignore):
    agent_executor = get_extension()
    output = list(agent_executor.stream({"input": query_widget.value}))
    with out:
        display(output[-1]["output"])


button = widgets.Button(description="Submit")
button.on_click(run_refresh)


def dialog():
    return display(query_widget, key_widget, button, out)
