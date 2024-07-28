import os
from ase.atoms import Atoms
from ase.build import bulk
from ase.constraints import UnitCellFilter
from ase.eos import calculate_eos, plot
from ase.optimize import LBFGS
from ase.units import kJ
import pandas
from langchain.agents import tool

from langsim.tools.helper import get_calculator
from langsim.tools.datatypes import AtomsDict
from langsim.tools.wolfram import download as wolframalpha_download


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
def get_atom_dict_equilibrated_structure(atom_dict: AtomsDict, calculator_str: str) -> AtomsDict:
    """
    Returns equilibrated atoms dictionary for a given bulk atoms dictionary and a selected model specified by the calculator string.

    Args:
        atom_dict (AtomsDict): DataClass representing the atomic structure
        calculator_str (str): selected model specified by the calculator string

    Returns:
        AtomsDict: DataClass representing the equilibrated atomic structure
    """
    try:
        atoms = Atoms(**atom_dict.dict())
        atoms.calc = get_calculator(calculator_str=calculator_str)
        ase_optimizer_obj = LBFGS(UnitCellFilter(atoms))
        ase_optimizer_obj.run(fmax=0.000001)
        return AtomsDict(**{k: v.tolist() for k, v in atoms.todict().items()})
    except Exception as error:
        # handle the exception
        return("An exception occurred: {}")


@tool
def plot_equation_of_state(atom_dict: AtomsDict, calculator_str: str) -> str:
    """
    Returns plot of equation of state of chemical symbol for a given equilibrated atoms dictionary and a selected model specified by the calculator string

    Args:
        atom_dict (AtomsDict): DataClass representing the atomic structure
        calculator_str (str): selected model specified by the calculator string

    Returns:
        str: plot of the equation of state
    """
    atoms = Atoms(**atom_dict.dict())
    atoms.calc = get_calculator(calculator_str=calculator_str)
    eos = calculate_eos(atoms)
    plotdata = eos.getplotdata()
    return plot(*plotdata)


@tool
def get_bulk_modulus(atom_dict: AtomsDict, calculator_str: str) -> str:
    """
    Returns the bulk modulus in GPa of chemical symbol for a given equilibrated atoms dictionary and a selected model specified by the calculator string

    Args:
        atom_dict (AtomsDict): DataClass representing the atomic structure
        calculator_str (str): selected model specified by the calculator string

    Returns:
        str: Bulk Modulus in GPa
    """
    try:
        atoms = Atoms(**atom_dict.dict())
        atoms.calc = get_calculator(calculator_str=calculator_str)
        eos = calculate_eos(atoms)
        v, e, B = eos.fit()
        return B / kJ * 1.0e24
    except Exception as error:
        # handle the exception
        return("An exception occurred: {}")


@tool
def get_equilibrium_volume(atom_dict: AtomsDict, calculator_str: str) -> str:
    """
    Returns the equilibrium volume in Angstrom^3 of chemical symbol for a given equilibrated atoms dictionary and a selected model specified by the calculator string

    Args:
        atom_dict (AtomsDict): DataClass representing the atomic structure
        calculator_str (str): selected model specified by the calculator string

    Returns:
        str: Equilibrium volume in Angstrom^3
    """
    atoms = Atoms(**atom_dict.dict())
    atoms.calc = get_calculator(calculator_str=calculator_str)
    eos = calculate_eos(atoms)
    v, e, B = eos.fit()
    return v


@tool
def get_experimental_elastic_property_wikipedia(chemical_symbol: str, property: str) -> str:
    """
    Looks up elastic properties for a given chemical symbol from the Wikipedia: https://en.wikipedia.org/wiki/Elastic_properties_of_the_elements_(data_page) sourced from webelements.com.

    Args:
        chemical_symbol (str): Chemical symbol of the element.
        property (str): Name of the property to retrieve. Options: youngs_modulus, poissons_ratio, bulk_modulus, shear_modulus

    Returns:
        str: Property value (various types): Value of the property for the given element, if available.
    """
    import pandas as pd
    
    tables=pd.read_html("https://en.wikipedia.org/wiki/Elastic_properties_of_the_elements_(data_page)")
    # Check if the property exists for the element
    try:
        property_options = {
            "youngs_modulus": [0, "GPa"],
            "poissons_ratio": [1, ""],
            "bulk_modulus": [2, "GPa"],
            "shear_modulus": [3, "GPa"],
        }
        lookup = property_options.get(property)
        lookup_id =lookup[0]
        unit = lookup[1]
        lookup_table = tables[lookup_id]
        # Take the column that extracts experimental value from 
        property_value = lookup_table[lookup_table['symbol'] == chemical_symbol]['WEL[1]'].item()
        return f"{property_value} {unit}"
    except:
        return f"Property '{property}' is not available for the element '{chemical_symbol}'."


@tool
def get_element_property_mendeleev(chemical_symbol: str, property: str) -> str:
    """
    Get the property of a given chemical symbol from the Mendeleev database.

    Args:
        chemical_symbol (str): Chemical symbol of the element.
        property (str): Name of the property to retrieve.

    Returns:
        str: Property (various types): Value and unit of the property for the given element, if available in Mendeleev database.
    """
    from mendeleev import element
    
    property_units = {
        'atomic_number': '',
        'symbol': '',
        'name': '',
        'atomic_mass': 'g/mol',
        'density': 'g/cm³',
        'melting_point': 'K',
        'boiling_point': 'K',
        'electronegativity': '',
        'ionenergies': 'eV',
        'electron_affinity': 'eV',
        'covalent_radius': 'Å',
        'vdw_radius': 'Å',
        'atomic_volume': 'cm³/mol',
        'poissons_ratio': '',
        'specific_heat_capacity': 'J/(g*K)',
        'thermal_conductivity': 'W/(m*K)',
        'electrical_resistivity': 'µΩ*cm'
    }

    elem = element(chemical_symbol)
    
    # Convert property name to lowercase for case insensitivity
    property = property.lower()

    # Check if the property exists for the element
    if hasattr(elem, property):
        property_value = getattr(elem, property)
        unit = property_units.get(property, '')
        if unit != "":
            return f"{property_value} {unit}"
        else:
            return str(property_value)
    else:
        return f"Property '{property}' is not available for the element '{chemical_symbol}'."


@tool
def get_chemical_information_from_wolframalpha(chemical_element):
    """
    Get information of a given chemical element

    Args:
        chemical_element: Chemical Element like Au for Gold

    Returns:
        dict: Dictionary with the following keys
            element: chemical element
            thermalcondictivity: thermal conductivity
            atomicradius: calculated distance from nucleus of outermost electron
            bulkmodulus: bulk modulus (incompressibility)
            shearmodulus: shear modulus of solid
            youngmodulus: Young's modulus of solid
            poissonratio: Poisson ratio of solid
            density: density at standard temperature and pressure
            liquiddensity: liquid density at melting point
            thermalexpansion: linear thermal expansion coefficient
            meltingpoint: melting temperature in kelvin
            vaporizationheat: latent heat for liquid-gas transition
            specificheat: specific heat capacity
            latticeconstant: crystal lattice constants
            crystal: basic crystal lattice structure
            volmolar: molar volume
            mass: average atomic weight in atomic mass units
            volume: Volume
    """
    filename = os.path.join(os.path.dirname(__file__), "..", "data", "wolfram.csv")
    if not os.path.exists(filename):
        wolframalpha_download()
    df = pandas.read_csv(filename)
    return df[df.element==chemical_element].squeeze(axis=0).to_dict()
