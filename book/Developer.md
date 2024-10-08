# For Developers
## Installation
All dependencies are available as `conda` packages via the `conda-forge` community channel, so just clone the repository
and install the environment file provided with the repository: 
```
git clone https://github.com/jan-janssen/LangSim
cd LangSim
conda env create -f environment.yml --name LangSim
conda activate LangSim
```
The [mace](https://mace-docs.readthedocs.io/en/latest/) - Machine Learning Force Fields - can be installed as optional dependency on Linux and MacOs using: 
```
conda install -c conda-forge pymace
```
The same applies to the `sqsgenerator` and `structuretoolkit`, which are required for the inverse alloy design example:
```
conda install -c conda-forge sqsgenerator structuretoolkit
```
Finally, `langsim` can be installed into the Python path using:
```
pip install .
```
The demonstration notebook is included in the `notebooks` folder `notebooks/demonstration.ipynb`.

Any feedback and suggestions are very appreciated, just [open an issue on Github](https://github.com/jan-janssen/LangSim).

## Benchmark
To benchmark different LLM providers we suggest configuring `LangSim` via environment variables:

For [Anthropic](http://anthropic.com):
```python
import os
os.environ["LANGSIM_PROVIDER"] = "anthropic" 
os.environ["LANGSIM_API_KEY"] = os.environ['ANTHROPIC_API_KEY']
os.environ["LANGSIM_MODEL"] = "claude-3-5-sonnet-20240620"
```

For [OpenAPI](https://openai.com):
```python
import os
os.environ["LANGSIM_API_KEY"] = os.environ['OPENAI_API_KEY']
os.environ["LANGSIM_MODEL"] = "gpt-4o"
```

Other providers, especially those who host open-source LLMs seem to struggle with LLM agents based on python functions.
So far only Anthropic and OpenAPI were successfully tested with `LangSim`. 

## Debugging
For analyzing the messages the [langchain](https://www.langchain.com) framework communicates to the LLM providers, you 
can enable the debug mode for the `langchain` package, using the following commands:  
```python
from langchain.globals import set_debug
set_debug(True)
```

## Known Limitations
The primary limitation at the moment is the restriction of the LLMs to a specific number of token, which restricts the 
number of python functions which can be implemented as simulation agents. A typical error message indicating this 
limitation would be: 
```
{
    'error': {
        'message': "Invalid 'tools[5].function.description': string too long. Expected a string with maximum length 1024, but got a string with length 1064 instead.", 
        'type': 'invalid_request_error', 
        'param': 'tools[5].function.description', 
        'code': 'string_above_max_length'
    }
}
```

Another known limitation is that the docstrings and system prompts have to be optimized for the different LLMs. Even for
LLMs from the same provider like ChatGPT 4.0 and ChatGPT 4.0o, both require different docstrings and system prompts. To 
illustrate this point, we created two branches [working_with_chatgpt4](https://github.com/jan-janssen/LangSim/tree/working_with_chatgpt4)
and [working_with_chatgpt4o](https://github.com/jan-janssen/LangSim/tree/working_with_chatgpt4o), which were optimized 
for ChatGPT 4.0 and ChatGPT 4.0o respectively. 