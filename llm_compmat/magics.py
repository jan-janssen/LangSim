"""
Magics to support LLM interactions in IPython/Jupyter.
"""

# Stdlib imports
import os
import sys

from base64 import b64decode

# from IPython/Jupyter APIs
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)

from IPython.core.magic_arguments import (magic_arguments, argument,
parse_argstring)

from IPython.display import display, Image, JSON, Markdown
from .llm import get_extension

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_output(messages):
    agent_executor = get_extension(OPENAI_API_KEY=OPENAI_API_KEY)
    return list(agent_executor.stream({"conversation": messages))[-1]

# Utility functions
MESSAGES = []

def get_chat_completion(prompt, temperature=0.6):
    MESSAGES.append(("user", prompt))
    return get_output(messages=MESSAGES)


def get_code_completion(instruction, input="", temperature=0.1):
    try:
        response = openai.Edit.create(
                    model="code-davinci-edit-001",
                    instruction=instruction,
                    input=input,
                    temperature=temperature,
                )
    except openai.InvalidRequestError as e:
        print(f"ERROR: {e.user_message}", file=sys.stderr)
    else:
        return response


def get_image(prompt, n=1, size="256x256"):
    response = openai.Image.create(prompt=prompt,
                                   n=n, size=size,
                                   response_format="b64_json")
    return response


# Class to manage state and expose the main magics

# The class MUST call this class decorator at creation time
@magics_class
class CompMatMagics(Magics):
    def __init__(self, shell):
        # You must call the parent constructor
        super(CompMatMagics, self).__init__(shell)
        #self.api_key = openai.api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.last_code = ""

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
        "Chat with GPTChat."
        args = parse_argstring(self.chat, line)

        if cell is None:
            prompt = ' '.join(args.prompt)
        else:
            prompt = cell
        response = get_chat_completion(prompt, temperature=args.temp)
        output = response['output']
        MESSAGES.append(("ai", output))
        #output = response.choices[0].text.strip()
        if args.raw:
            return Markdown(f"```\n{output}\n```\n")
        else:
            return Markdown(output)

    @magic_arguments()
    @argument(
        '-l', '--lang', default='python',
        help="""Language, used only for formatting the output, you'll need to specify the language in the prompt for ChatGPT to understand. [Default: python].
        """
    )
    @argument('-s', '--sep', type=str, default="##",
        help="""Separator between instruction prompt and code input 
        [default: ##].""")
    @argument('-T', '--temp', type=float, default=0.1,
        help="""Temperature, float in [0,1]. Higher values push the algorithm
        to generate more aggressive/"creative" output. [default=0.1].""")
    @argument('prompt', nargs='*',
        help="""Prompt for code generation. When used as a line magic,
        it runs to the end of the line. In cell mode, the entire cell
        is considered the code generation prompt.
        """)
    @line_cell_magic
    def code(self, line, cell=None):
        """Prompt for code generation.

        The prompt can be purely an instruction (in natural language), or it can consist of both an instruction and a block of code. If you want to provide code, it should come below the instruction, after a separator (which defaults to `##` but can be set with the `--sep` argumet).

        The code is displayed as syntax-highlighted Markdown. If you want to get the code directly inserted into a cell for later execution, call the `%get_code` magic.

        Example:

        ```
        %%code

        Please add a docstring to this function.

        ##

        def add(a, b): return a+b
        ```
        """
        args = parse_argstring(self.code, line)
        input = ""
        if cell is None:
            instruction = ' '.join(args.prompt)
        else:
            parts =  cell.split(args.sep)
            if len(parts) == 1:
                instruction = cell
            elif len(parts) == 2:
                instruction, input = parts
            else:
                print("Only one ## marker is supported", file=sys.stderr)
                return
        if not(input) and "CODE" in instruction:
            input = self.last_code
        
        response = get_code_completion(instruction, input, 
                                       temperature=args.temp)
        if response:
            self.last_code = response.choices[0].text.strip()
            return Markdown(f"```{args.lang}\n{self.last_code}\n```")
        else:
            print("ERROR: No response returned by GPT. Please try again.",
                  file=sys.stderr)

    @line_magic
    def get_code(self, line):
        """
        Insert the last code block created with `%code` into a new code cell.
        """
        self.shell.set_next_input(self.last_code)

    @magic_arguments()
    @argument(
        '-s', '--size', choices=['s', 'm', 'l'], default='s',
        help="""Image size (s=256x256, m=512x512, l=1024x1024). [Default=s]."""
    )
    @argument('-n', type=int, default=1,
        help="Number of images to generate [default=1].")
    @argument('prompt', nargs='*',
        help="""Prompt for image generation. When used as a line magic,
        it runs to the end of the line. In cell mode, the entire cell
        is considered the image generation prompt.
        """)
    @line_cell_magic
    def pic(self, line, cell=None):
        """Prompt for image generation.
        """
        args = parse_argstring(self.pic, line)
        prompt = ' '.join(args.prompt) if cell is None else cell
        sizes = {'s':"256x256", 'm':"512x512", 'l':"1024x1024"}

        response = get_image(prompt, args.n, sizes[args.size])
        for img in response.data:
            display(Image(data=b64decode(img.b64_json)))


# If testing interactively, it's convenient to %run as a script in Jupyter
if __name__ == "__main__":
    get_ipython().register_magics(CompMatMagics)
