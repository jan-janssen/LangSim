"""
Magics to support LLM interactions in IPython/Jupyter.

Adapted from fperez/jupytee
"""

# Stdlib imports
import os

from IPython import get_ipython
from IPython.core.magic import (
    Magics,
    magics_class,
    line_cell_magic,
)
from IPython.core.magic_arguments import (
    magic_arguments,
    argument,
    parse_argstring,
)
from IPython.display import Markdown
from langsim.llm import get_executor


def get_output(messages):
    env = os.environ
    agent_executor = get_executor(
        api_provider=env.get("LANGSIM_PROVIDER", "OPENAI"),
        api_key=env.get("LANGSIM_API_KEY"),
        api_url=env.get("LANGSIM_API_URL", None),
        api_model=env.get("LANGSIM_MODEL", None),
        api_temperature=env.get("LANGSIM_TEMP", 0),
    )
    return list(agent_executor.stream({"conversation": messages}))[-1]


# Class to manage state and expose the main magics
# The class MUST call this class decorator at creation time
@magics_class
class CompMatMagics(Magics):
    def __init__(self, shell):
        # You must call the parent constructor
        super(CompMatMagics, self).__init__(shell)
        self.last_code = ""
        self.messages = []

    @magic_arguments()
    @argument(
        '-r', '--raw', action="store_true",
        help="""Return output as raw text instead of rendering it as Markdown[Default: False].
        """
    )
    @argument('-T', '--temp', type=float, default=0.6,
        help="""Temperature, float in [0,1]. Higher values push the algorithm
        to generate more aggressive/"creative" output. [default=0.1].""")
    @argument('prompt', nargs='*',
        help="""Prompt for code generation. When used as a line magic,
        it runs to the end of the line. In cell mode, the entire cell
        is considered the code generation prompt.
        """)
    @line_cell_magic
    def chat(self, line, cell=None):
        """
        Chat with GPTChat.
        """
        args = parse_argstring(self.chat, line)

        if cell is None:
            prompt = ' '.join(args.prompt)
        else:
            prompt = cell
        self.messages.append(("human", prompt))
        response = get_output(self.messages)
        output = response['output']
        self.messages.append(("ai", output))
        if args.raw:
            return Markdown(f"```\n{output}\n```\n")
        elif isinstance(output, list) and isinstance(output[-1], dict):
            return Markdown(output[-1]['text'])
        else:
            return Markdown(output)


# If testing interactively, it's convenient to %run as a script in Jupyter
if __name__ == "__main__":
    get_ipython().register_magics(CompMatMagics)
