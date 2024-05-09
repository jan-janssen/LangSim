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
def get_equilibrium_lattice(chemical_symbol: str, calculator_str: str) -> AtomsDict:
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


@tool
def get_experimental_elastic_property_wikipedia(chemical_symbol: str, property: str) -> str:
    """
    Looks up elastic properties for a given chemical symbol from the Wikipedia: https://en.wikipedia.org/wiki/Elastic_properties_of_the_elements_(data_page) sourced from webelements.com.

    Args:
    - symbol (str): Chemical symbol of the element.
    - property_name (str): Name of the property to retrieve. Options: youngs_modulus, poissons_ratio, bulk_modulus, shear_modulus 

    Returns:
    - Property value (various types): Value of the property for the given element, if available.
    """
    import pandas as pd
    
    tables=pd.read_html("https://en.wikipedia.org/wiki/Elastic_properties_of_the_elements_(data_page)")
    # Check if the property exists for the element
    try:
        property_options = {"youngs_modulus":[0,"GPa"], "poissons_ratio":[1,""], "bulk_modulus":[2,"GPa"], "shear_modulus":[3, "GPa"]}
        lookup = property_options.get(property)
        lookup_id =lookup[0]
        unit = lookup[1]
        lookup_table = tables[lookup_id]
        # Take the column that extracts experimental value from 
        property_value = lookup_table[lookup_table['symbol']==chemical_symbol]['WEL[1]'].item()
        return f"{property_value} {unit}"
    except:
        return f"Property '{property}' is not available for the element '{chemical_symbol}'."


@tool
def get_element_property_mendeleev(chemical_symbol: str, property: str) -> str:
    """
    Get the property of a given chemical symbol from the Mendeleev database.

    Args:
    - symbol (str): Chemical symbol of the element.
    - property_name (str): Name of the property to retrieve. 

    Returns:
    - Property (various types): Value and unit of the property for the given element, if available in Mendeleev database.
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
        return f"{property_value} {unit}"
    else:
        return f"Property '{property}' is not available for the element '{chemical_symbol}'."