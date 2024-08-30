import os
from ase.atoms import Atoms
from ase.build import bulk
from ase.eos import calculate_eos, plot
from ase.optimize import LBFGS
from atomistics.calculators import evaluate_with_ase
from atomistics.workflows import optimize_positions_and_volume
from atomistics.workflows.evcurve.helper import (
    analyse_structures_helper,
    generate_structures_helper,
)
import pandas
from langchain.agents import tool
from mendeleev.fetch import fetch_table

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
        return("An exception occurred: {}")


@tool
def plot_equation_of_state(atom_dict: AtomsDict, calculator_str: str) -> str:
    """
    Returns plot of equation of state of chemical symbol for a given equilibrated atoms dictionary and a selected model specified by the calculator string

    Args:
        atom_dict (AtomsDict): DataClass representing the atomic structure
        calculator_str (str): a model from the following options: "emt", "mace"
                              - "emt": Effective medium theory is a computationally efficient, analytical model
                                 that describes the macroscopic properties of composite materials. 
                              - "mace": this is a machine learning force field for predicting many-body atomic
                                 interactions that covers the periodic table.     

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
        return("An exception occurred: {}")


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
def get_chemical_information_from_mendeleev(chemical_symbol: str) -> dict:
    """
    Get information of a given chemical element

    Args:
        chemical_symbol: Chemical Element like Au for Gold

    Returns:
        dict: Dictionary with the following keys
            abundance_crust: Abundance in the Earth’s crust in mg/kg
            abundance_sea: Abundance in the seas in mg/L
            atomic_number: Atomic number
            atomic_radius_rahm: Atomic radius by Rahm et al. in pm
            atomic_radius: Atomic radius in pm
            atomic_volume: Atomic volume in cm^3/mol
            atomic_weight_uncertainty: Atomic weight uncertainty in Da
            atomic_weight: Relative atomic weight in Da
            block: Block in periodic table
            boiling_point: Boiling point in K
            c6_gb: C_6 dispersion coefficient according to Gould & Bučko in hartree/bohr^6
            c6: C_6 dispersion coefficient in hartree/bohr^6
            cas: Chemical Abstracts Serice identifier
            covalent_radius_bragg: Covalent radius by Bragg in pm
            covalent_radius_cordero: Covalent radius by Cerdero et al. in pm
            covalent_radius_pyykko_double: Double bond covalent radius by Pyykko et al. in pm
            covalent_radius_pyykko_triple: Triple bond covalent radius by Pyykko et al. in pm
            covalent_radius_pyykko: Single bond covalent radius by Pyykko et al. in pm
            cpk_color: Element color in CPK convention
            critical_pressure: Critical pressure in MPa
            critical_temperature: Critical temperature in K
            density: Density at 295K in g/cm^3
            description: Short description of the element
            dipole_polarizability_unc: Uncertainty of the dipole polarizability in bohr^3
            dipole_polarizability: Dipole polarizability in bohr^3
            discoverers: The discoverers of the element
            discovery_location: The location where the element was discovered
            discovery_year: The year the element was discovered
            econf: Ground state electronic configuration
            electron_affinity: Electron affinity in eV
            electronegativity_allen: Allen’s scale of electronegativity in eV
            electronegativity_allred_rochow: Allred and Rochow’s scale of electronegativity in e^2/pm^2
            electronegativity_cottrell_sutton: Cottrell and Sutton’s scale of electronegativity in e^0.5/pm^0.5
            electronegativity_ghosh: Ghosh’s scale of electronegativity in 1/pm
            electronegativity_gordy: Gordy’s scale of electronegativity in e/pm
            electronegativity_li_xue: Li and Xue’s scale of electronegativity in 1/pm
            electronegativity_martynov_batsanov: Martynov and Batsanov’s scale of electronegativity in eV^0.5
            electronegativity_mulliken: Mulliken’s scale of electronegativity in eV
            electronegativity_nagle: Nagle’s scale of electronegativity in 1/bohr
            electronegativity_pauling: Pauling’s scale of electronegativity
            electronegativity_sanderson: Sanderson’s scale of electronegativity
            electrons: Number of electrons
            electrophilicity: Parr’s electrophilicity index
            evaporation_heat: Evaporation heat in kJ/mol
            fusion_heat: Fusion heat in kJ/mol
            gas_basicity: Gas basicity in kJ/mol
            geochemical_class: Geochemical classification
            glawe_number: Glawe’s number (scale)
            goldschmidt_class: Goldschmidt classification
            group: Group in the periodic table
            hardness: Absolute hardness. Can also be calcualted for ions. in eV
            heat_of_formation: Heat of formation in kJ/mol
            inchi: International Chemical Identifier
            ionenergy: See IonizationEnergy class documentation
            ionic_radii: See IonicRadius class documentation
            is_monoisotopic: Is the element monoisotopic
            is_radioactive: Is the element radioactive
            isotopes: See Isotope class documentation
            jmol_color: Element color in Jmol convention
            lattice_constant: Lattice constant
            lattice_structure: Lattice structure code
            mass_number: Mass number of the most abundant isotope
            melting_point: Melting point in K
            mendeleev_number: Mendeleev’s number
            metallic_radius_c12: Metallic radius with 12 nearest neighbors in pm
            metallic_radius: Single-bond metallic radius in pm
            molar_heat_capacity: Molar heat capacity @ 25 C, 1 bar in J/mol/K
            molcas_gv_color: Element color in MOCAS GV convention
            name_origin: Origin of the name
            name: Name in English
            neutrons: Number of neutrons
            nist_webbook_url: URL for the NIST Chemistry WebBook
            nvalence: Number of valence electrons
            oxides: Possible oxides based on oxidation numbers
            oxistates: See OxidationState class documentation
            period: Period in periodic table
            pettifor_number: Pettifor scale
            proton_affinity: Proton affinity in kJ/mol
            protons: Number of protons
            sconst: See ScreeningConstant class documentation
            series: Series in the periodic table
            softness: Absolute softness. Can also be calculated for ions. in 1/eV
            sources: Sources of the element
            specific_heat_capacity: Specific heat capacity @ 25 C, 1 bar in J/g/K
            symbol: Chemical symbol
            thermal_conductivity: Thermal conductivity @25 C in W/m/K
            triple_point_pressure: Presseure of the triple point in kPa
            triple_point_temperature: Temperature of the triple point in K
            uses: Main applications of the element
            vdw_radius_alvarez: Van der Waals radius according to Alvarez in pm
            vdw_radius_batsanov: Van der Waals radius according to Batsanov in pm
            vdw_radius_bondi: Van der Waals radius according to Bondi in pm
            vdw_radius_dreiding: Van der Waals radius from the DREIDING FF in pm
            vdw_radius_mm3: Van der Waals radius from the MM3 FF in pm
            vdw_radius_rt: Van der Waals radius according to Rowland and Taylor in pm
            vdw_radius_truhlar: Van der Waals radius according to Truhlar in pm
            vdw_radius_uff: Van der Waals radius from the UFF in pm
            vdw_radius: Van der Waals radius in pm
            zeff: Effective nuclear charge
    """
    df = fetch_table('elements')
    return df[df.symbol == chemical_symbol].squeeze(axis=0).to_dict()
 

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
