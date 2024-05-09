from setuptools import setup, find_packages

setup(
    name='llm_compmat',
    version='0.0.1',
    url='https://github.com/jan-janssen/comp-mat-sci-llm',
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