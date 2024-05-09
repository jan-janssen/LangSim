import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

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


def get_plot(output_steps):
    for step in output_steps:
        if hasattr(step, 'observation') and isinstance(step.observation, plt.Axes):
            return step.observation


def run_refresh(*ignore):
    agent_executor = get_extension(OPENAI_API_KEY=key_widget.value)
    output = list(agent_executor.stream({"input": query_widget.value}))
    with out:
        for part in output:
            if "steps" in part.keys():
                plot_axes = get_plot(part["steps"])
                if isinstance(plot_axes, plt.Axes):
                    display(plot_axes.figure)
    display(output[-1]["output"])


button = widgets.Button(description="Submit")
button.on_click(run_refresh)


def dialog():
    return display(query_widget, key_widget, button, out)
