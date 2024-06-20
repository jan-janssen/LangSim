# Run Calculations with a Large Language Model
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jan-janssen/LangSim/HEAD?labpath=notebooks/demonstration.ipynb)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/ltalirz/LangSim)

The computational chemistry and computational materials science community have both developed a great number of 
simulation tools. Still these tools typically require either rather cryptic input files or at least a fundamental 
programming experience in a language like Python to control them. Furthermore, many questions are only answered in the 
documentation, like: 
* Which physical units does the code use? 
* Which inputs match to which variables in the equations in the paper? 
* ...

We address this challenge by developing a Large Language Model (LLM) extension which provides LLM agents to couple the 
LLM to scientific simulation codes and calculate physical properties from a natural language interface.

![Demonstration](docs/images/demonstration.gif)

## Installation 
### Via pip 
While our package is not yet available on the Python Package Index, you can install it directly using:
```
pip install git+https://github.com/jan-janssen/LangSim.git
```
The pip package includes optional dependencies for the `mace` model and the `jupyter` integration.  

### Via conda 
As the conda package is not yet available on Anaconda.org still you can clone the repository and install the 
dependencies directly from conda using the [environment.yml](environment.yml) file. 

Prerequisites:
- [git](https://git-scm.com/)
- [conda](https://docs.conda.io/en/latest/miniconda.html)

 ```bash
git clone https://github.com/jan-janssen/LangSim
cd LangSim
conda env create -f environment.yml --name LangSim
```

### Via Docker Container
We build a docker container on every commit to the `main` branch.
You can pull the container from the [Docker Hub](https://hub.docker.com/r/ltalirz/langsim) using:
```bash
docker run -p 8866:8866 ltalirz/langsim
```

## Using the package
The package currently provides two interfaces, one for python / jupyter users to query the large language model directly
from a python environment and a second web based interface. 

## Contributors
* [Yuan Chiang](https://github.com/chiang-yuan)
* [Giuseppe Fisicaro](https://github.com/giuseppefisicaro)
* [Jan Janssen](https://github.com/jan-janssen)
* [Greg Juhasz](https://github.com/gjuhasz)
* [Sarom Leang](https://github.com/saromleang)
* [Bernadette Mohr](https://github.com/Bernadette-Mohr)
* [Utkarsh Pratiush](https://github.com/utkarshp1161)
* [Francesco Ricci](https://github.com/fraricci)
* [Leopold Talirz](https://github.com/ltalirz)
* [Pablo Andres Unzueta](https://github.com/pablo-unzueta)
* [Trung Vo](https://github.com/btrungvo)
* [Gabriel Vogel](https://github.com/GaVogel)
* [Luke Zondagh](https://github.com/Luke-Zondagh)
* [Sebastian Pagel](https://github.com/pagel-s)
