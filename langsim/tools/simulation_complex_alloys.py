from typing import List

from ase.atoms import Atoms
from ase.build import bulk
from ase.data import reference_states, atomic_numbers
from langchain.tools import tool
import numpy as np
import structuretoolkit as stk

from langsim.tools.simulation_atomistics import AtomsDict


def _get_element_dict(element_lst: List[str], concentration_lst: List[float], number_of_atoms: int) -> dict:
    element_dict = {
        el: int(np.round(c * number_of_atoms))
        for el, c in zip(element_lst[:-1], concentration_lst[:-1])
    }
    element_dict[element_lst[-1]] = number_of_atoms-sum(element_dict.values())
    return element_dict


def _get_alloy_concentration(element_lst: List[str], concentration_lst: List[float], number_of_atoms: int) -> dict:
    if sum(concentration_lst) > 1:
        raise ValueError()
    elif sum(concentration_lst) < 1 and len(concentration_lst)+1 == len(element_lst):
        concentration_lst.append(1-sum(concentration_lst))
        return _get_element_dict(
            element_lst=element_lst,
            concentration_lst=concentration_lst,
            number_of_atoms=number_of_atoms,
        )
    elif sum(concentration_lst) == 1 and len(concentration_lst) == len(element_lst):
        return _get_element_dict(
            element_lst=element_lst,
            concentration_lst=concentration_lst,
            number_of_atoms=number_of_atoms,
        )
    else:
        raise ValueError()


def _get_nn_distance(symmetry: str, a: float, **kwargs) -> float:
    if symmetry.lower() == "fcc":
        return (a * np.sqrt(2))/2
    elif symmetry.lower() == "bcc":
        return (a * np.sqrt(3))/4
    else:
        raise ValueError()


def _get_lattice_constant(symmetry: str, nn_distance: float) -> float:
    if symmetry.lower() == "fcc":
        return (2 * nn_distance) / np.sqrt(2)
    elif symmetry.lower() == "bcc":
        return (4 * nn_distance) / np.sqrt(3)
    else:
        raise ValueError()


def _get_template(crystal_structure: str, a:float, number_of_atoms: int):
    if crystal_structure.lower() == "fcc":
        structure = bulk("Al", a=a, cubic=True)
    elif crystal_structure.lower() == "bcc":
        structure = bulk("Fe", a=a, cubic=True)
    else:
        raise ValueError()

    i = 1
    structure_template = structure.copy()
    while len(structure) < number_of_atoms:
        structure = structure_template.repeat([i, i, i])
        i+=1
    return structure


def _get_structure(element_lst: List[str], concentration_lst: List[float], number_of_atoms: int, crystal_structure:str) -> Atoms:
    number_of_atoms= len(_get_template(crystal_structure=crystal_structure, a=1.0, number_of_atoms=number_of_atoms))
    if len(element_lst) == 1 + sum([c == 0 for c in concentration_lst]):
        return _get_template(
            crystal_structure=crystal_structure,
            a=reference_states[atomic_numbers[element_lst[0]]]["a"],
            number_of_atoms=number_of_atoms
        )
    else:
        element_dict = _get_alloy_concentration(
            element_lst=element_lst,
            concentration_lst=concentration_lst,
            number_of_atoms=number_of_atoms,
        )
        a = _get_lattice_constant(
            symmetry=crystal_structure,
            nn_distance=sum({
                el: n * _get_nn_distance(**reference_states[atomic_numbers[el]]) / number_of_atoms
                for el, n in element_dict.items()
            }.values()),
        )
        structure_template = _get_template(crystal_structure=crystal_structure, a=a, number_of_atoms=number_of_atoms)
        structure_template.set_chemical_symbols([list(element_dict.keys())[0]] * number_of_atoms)
        return stk.build.sqs_structures(
            structure=structure_template,
            mole_fractions={el: n/ number_of_atoms for el, n in element_dict.items()},
            objective=0.0,
            iterations=1e6,
            output_structures=1,
        )[0]


@tool
def get_complex_alloy_bulk_structure(element_lst: List[str], concentration_lst: List[float], number_of_atoms: int = 32, crystal_structure:str = "fcc") -> AtomsDict:
    """
    Returns bulk structure atoms dictionary for a given list of chemical elements and their corresponding concentration

    Args:
        element_lst (List[str]): list of chemical elements
        concentration_lst (List[float]): list of concentrations of the corresponding chemical elements
        number_of_atoms (int): number of atoms in the structure - optional - by default 32
        crystal_structure (str): atomic crystal structure of the solid solution - by default "fcc" - face centered cubic

    Returns:
        AtomsDict: DataClass representing the atomic structure required to equilibrate the structure
    """
    atoms = _get_structure(element_lst=element_lst, concentration_lst=concentration_lst, number_of_atoms=number_of_atoms, crystal_structure=crystal_structure)
    return AtomsDict(**{k: v.tolist() for k, v in atoms.todict().items()})
