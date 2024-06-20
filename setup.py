from setuptools import setup, find_packages

setup(
    name='llm_compmat',
    version='0.0.1',
    url='https://github.com/jan-janssen/comp-mat-sci-llm',
    description='Application of Large Language Models (LLM) for computational materials science',
    packages=find_packages(),
    install_requires=[
        "ase==3.23.0",
        "langchain==0.2.5",
        "numexpr==2.10.0",
        "langchain-openai==0.1.8",
        "langchain-experimental==0.0.61",
        "mace=0.3.5",
        "matplotlib==3.8.4",
        "pandas==2.2.2",
        "lxml==4.9.4",
        "mendeleev==0.17.0"
    ],
    extra_requires={
        "mace": ["mace"],
        "jupyter": ["voila", "ipykernel", "ipywidgets",]
    }
)