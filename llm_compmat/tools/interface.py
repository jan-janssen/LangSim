from ase.atoms import Atoms
from ase.build import bulk
from ase.constraints import UnitCellFilter
from ase.eos import calculate_eos, plot
from ase.optimize import LBFGS
from ase.units import kJ
from langchain.agents import tool

from llm_compmat.tools.helper import get_calculator
from llm_compmat.tools.datatypes import AtomsDict


@tool
def get_equilibirum_lattice(chemical_symbol: str, calculator_str: str) -> AtomsDict:
    """Returns equilibrium atoms dictionary for a given chemical symbol and a selected model specified by the calculator string"""
    atoms = bulk(name=chemical_symbol)
    atoms.calc = get_calculator(calculator_str=calculator_str)
    ase_optimizer_obj = LBFGS(UnitCellFilter(atoms))
    ase_optimizer_obj.run(fmax=0.000001)
    return AtomsDict(**{k: v.tolist() for k, v in atoms.todict().items()})


@tool
def plot_equation_of_state(atom_dict: AtomsDict, calculator_str: str) -> str:
    """Returns plot of equation of state of chemical symbol for a given atoms dictionary and a selected model specified by the calculator string"""
    atoms = Atoms(**atom_dict.dict())
    atoms.calc = get_calculator(calculator_str=calculator_str)
    eos = calculate_eos(atoms)
    plotdata = eos.getplotdata()
    return plot(*plotdata)


@tool
def get_bulk_modulus(atom_dict: AtomsDict, calculator_str: str) -> str:
    """Returns the bulk modulus of chemcial symbol for a given atoms dictionary and a selected model specified by the calculator string in GPa"""
    atoms = Atoms(**atom_dict.dict())
    atoms.calc = get_calculator(calculator_str=calculator_str)
    eos = calculate_eos(atoms)
    v, e, B = eos.fit()
    return B / kJ * 1.0e24


@tool
def get_equilibrium_volume(atom_dict: AtomsDict, calculator_str: str) -> str:
    """Returns the equilibrium volume of chemcial symbol for a given atoms dictionary and a selected model specified by the calculator string in Angstrom^3"""
    atoms = Atoms(**atom_dict.dict())
    atoms.calc = get_calculator(calculator_str=calculator_str)
    eos = calculate_eos(atoms)
    v, e, B = eos.fit()
    return v
