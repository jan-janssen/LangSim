# Developer Instructions
## Installation
All dependencies are available as `conda` packages via the `conda-forge` community channel, so just clone the repository
and install the environment file provided with the repository: 
```
git clone https://github.com/jan-janssen/LangSim
cd LangSim
conda env create -f environment.yml --name LangSim
conda activate LangSim
```
The [mace](https://mace-docs.readthedocs.io/en/latest/) - Machine Learning Force Fields can be installed as optional 
dependency on Linux and MacOs using: 
```
conda install -c conda-forge pymace
```
Finally, `langsim` can be installed into the Python path using:
```
pip install .
```
The demonstration notebook is included in the `notebooks` folder `notebooks/demonstration.ipynb`.

Any feedback and suggestions are very welcome, just [open an issue on Github](https://github.com/jan-janssen/LangSim). 

## Debugging
For analysing the messages the [langchain](https://www.langchain.com) framework communicates to the LLM providers, you 
can enable the debug mode for the `langchain` package:  
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