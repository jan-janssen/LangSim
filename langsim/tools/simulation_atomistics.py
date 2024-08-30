from typing import List

from ase.atoms import Atoms
from ase.build import bulk
from ase.optimize import LBFGS
from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import optimize_positions_and_volume
from atomistics.workflows.evcurve.helper import (
    analyse_structures_helper,
    generate_structures_helper,
)
from langchain.agents import tool
from langchain_core.pydantic_v1 import BaseModel


class AtomsDict(BaseModel):
    numbers: List[int]
    positions: List[List[float]]
    cell: List[List[float]]
    pbc: List[bool]


@tool
def get_atom_dict_bulk_structure(chemical_symbol: str) -> AtomsDict:
    """
    Returns bulk structure atoms dictionary for a given chemical symbol

    Args:
        chemical_symbol (str): chemical symbol as string

    Returns:
        AtomsDict: DataClass representing the atomic structure
    """
    atoms = bulk(name=chemical_symbol)
    return AtomsDict(**{k: v.tolist() for k, v in atoms.todict().items()})


@tool
def get_atom_dict_equilibrated_structure(
    atom_dict: AtomsDict, calculator_str: str
) -> AtomsDict:
    """
    Returns equilibrated atoms dictionary for a given bulk atoms dictionary and a selected model specified by the calculator string.

    Args:
        atom_dict (AtomsDict): DataClass representing the atomic structure
        calculator_str (str): a model from the following options: "emt", "mace"
                              - "emt": Effective medium theory is a computationally efficient, analytical model
                                 that describes the macroscopic properties of composite materials.
                              - "mace": this is a machine learning force field for predicting many-body atomic
                                 interactions that covers the periodic table.

    Returns:
        AtomsDict: DataClass representing the equilibrated atomic structure
    """
    try:
        task_dict = optimize_positions_and_volume(structure=Atoms(**atom_dict.dict()))
        atoms = evaluate_with_ase(
            task_dict=task_dict,
            ase_calculator=get_calculator(calculator_str=calculator_str),
            ase_optimizer=LBFGS,
            ase_optimizer_kwargs={"fmax": 0.000001},
        )["structure_with_optimized_positions_and_volume"]
        return AtomsDict(**{k: v.tolist() for k, v in atoms.todict().items()})
    except Exception as error:
        # handle the exception
        return "An exception occurred: {}"


@tool
def get_bulk_modulus(atom_dict: AtomsDict, calculator_str: str) -> str:
    """
    Returns the bulk modulus in GPa of chemical symbol for a given equilibrated atoms dictionary and a selected model specified by the calculator string

    Args:
        atom_dict (AtomsDict): DataClass representing the atomic structure
        calculator_str (str): a model from the following options: "emt", "mace"
                              - "emt": Effective medium theory is a computationally efficient, analytical model
                                 that describes the macroscopic properties of composite materials.
                              - "mace": this is a machine learning force field for predicting many-body atomic
                                 interactions that covers the periodic table.

    Returns:
        str: Bulk Modulus in GPa
    """
    try:
        structure_dict = generate_structures_helper(
            structure=Atoms(**atom_dict.dict()),
            vol_range=0.05,
            num_points=11,
            strain_lst=None,
            axes=("x", "y", "z"),
        )
        result_dict = evaluate_with_ase(
            task_dict={"calc_energy": structure_dict},
            ase_calculator=get_calculator(calculator_str=calculator_str),
        )
        fit_dict = analyse_structures_helper(
            output_dict=result_dict,
            structure_dict=structure_dict,
            fit_type="polynomial",
            fit_order=5,
        )
        return fit_dict["bulkmodul_eq"]
    except Exception as error:
        # handle the exception
        return "An exception occurred: {}"


@tool
def get_equilibrium_volume(atom_dict: AtomsDict, calculator_str: str) -> str:
    """
    Returns the equilibrium volume in Angstrom^3 of chemical symbol for a given equilibrated atoms dictionary and a selected model specified by the calculator string

    Args:
        atom_dict (AtomsDict): DataClass representing the atomic structure
        calculator_str (str): a model from the following options: "emt", "mace"
                              - "emt": Effective medium theory is a computationally efficient, analytical model
                                 that describes the macroscopic properties of composite materials.
                              - "mace": this is a machine learning force field for predicting many-body atomic
                                 interactions that covers the periodic table.

    Returns:
        str: Equilibrium volume in Angstrom^3
    """
    structure_dict = generate_structures_helper(
        structure=Atoms(**atom_dict.dict()),
        vol_range=0.05,
        num_points=11,
        strain_lst=None,
        axes=("x", "y", "z"),
    )
    result_dict = evaluate_with_ase(
        task_dict={"calc_energy": structure_dict},
        ase_calculator=get_calculator(calculator_str=calculator_str),
    )
    fit_dict = analyse_structures_helper(
        output_dict=result_dict,
        structure_dict=structure_dict,
        fit_type="polynomial",
        fit_order=5,
    )
    return fit_dict["volume_eq"]


def get_calculator(calculator_str: str = "emt"):
    if calculator_str == "emt":
        from ase.calculators.emt import EMT

        return EMT()
    elif calculator_str == "mace":
        from mace.calculators import mace_mp

        return mace_mp(
            model="medium", dispersion=False, default_dtype="float32", device="cpu"
        )
