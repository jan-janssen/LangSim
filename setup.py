from setuptools import setup, find_packages

setup(
    name='langsim',
    version='0.0.1',
    url='https://github.com/jan-janssen/LangSim',
    description='Application of Large Language Models (LLM) for computational materials science',
    packages=find_packages(),
    install_requires=[
        "ase==3.25.0",
        "atomistics==0.2.5",
        "langchain==0.3.26",
        "numexpr==2.10.2",
        "langchain-openai==0.3.27",
        "matplotlib==3.10.3",
        "pandas==2.3.1",
        "lxml==5.4.0",
        "mendeleev==0.19.0",
    ],
    extras_require={
        "mace": ["mace==0.3.5"],
        "alloy": [
            "sqsgenerator==0.3",
            "structuretoolkit==0.0.32",
        ],
        "ipython": [
            "ipython==8.25.0",
            "langchain-experimental==0.3.4",
        ],
    }
)
