from setuptools import setup, find_packages

setup(
    name='langsim',
    version='0.0.1',
    url='https://github.com/jan-janssen/LangSim',
    description='Application of Large Language Models (LLM) for computational materials science',
    packages=find_packages(),
    install_requires=[
        "ase",
        "langchain",
        "numexpr",
        "langchain-openai",
        "langchain-experimental",
        "mace",
        "matplotlib",
        "langchainhub",
        "pandas",
        "lxml",
        "mendeleev"
    ],
    extra_requires={
        "mace": ["mace"],
        "jupyter": ["voila", "ipykernel", "ipywidgets",]
    }
)