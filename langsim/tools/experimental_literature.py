from langchain_core.tools import tool


@tool
def get_experimental_elastic_property_wikipedia(chemical_symbol: str) -> dict:
    """
    Looks up elastic properties for a given chemical symbol from the Wikipedia: https://en.wikipedia.org/wiki/Elastic_properties_of_the_elements_(data_page) sourced from webelements.com.

    Args:
        chemical_symbol (str): Chemical symbol of the element.

    Returns:
        str: Property value (various types): Value of the property for the given element, if available.
    """
    from atomistics.referencedata import (
        get_elastic_properties_from_wikipedia as get_wikipedia,
    )

    return get_wikipedia(chemical_symbol=chemical_symbol)


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
    from atomistics.referencedata import (
        get_chemical_information_from_mendeleev as get_mendeleev,
    )

    return get_mendeleev(chemical_symbol=chemical_symbol)


@tool
def get_chemical_information_from_wolframalpha(chemical_symbol: str) -> dict:
    """
    Get information of a given chemical element

    Args:
        chemical_symbol: Chemical Element like Au for Gold

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
    from atomistics.referencedata import (
        get_chemical_information_from_wolframalpha as get_wolframalpha,
    )

    return get_wolframalpha(chemical_symbol=chemical_symbol)
