from unittest import TestCase
import numpy as np
from langsim.tools.simulation_atomistics import (
    get_atom_dict_bulk_structure,
    get_atom_dict_equilibrated_structure,
    get_bulk_modulus,
    get_equilibrium_volume,
)
from langsim.tools.experimental_literature import (
    get_chemical_information_from_mendeleev,
    get_experimental_elastic_property_wikipedia,
)


class TestInterfaceEMT(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculator_str = "emt"
        atom_dict = get_atom_dict_bulk_structure.invoke({"chemical_symbol": "Al"})
        cls.structure = get_atom_dict_equilibrated_structure.invoke(
            {"atom_dict": atom_dict, "calculator_str": cls.calculator_str}
        )

    def test_structure(self):
        self.assertTrue(np.all(np.isclose(self.structure.positions, [[0.0, 0.0, 0.0]])))
        self.assertEqual(self.structure.numbers, [13])
        self.assertTrue(all(self.structure.pbc))
        self.assertAlmostEqual(np.linalg.det(self.structure.cell), 15.931389072530033)

    def test_bulk_modulus(self):
        b0 = get_bulk_modulus.invoke({"atom_dict": self.structure, "calculator_str": self.calculator_str})
        self.assertAlmostEqual(b0, 39.73699875059587)

    def test_equilibrium_volume(self):
        v0 = get_equilibrium_volume.invoke({"atom_dict": self.structure, "calculator_str": self.calculator_str})
        self.assertAlmostEqual(v0, 15.931368308723524)


class TestInterfaceMace(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculator_str = "mace"
        atom_dict = get_atom_dict_bulk_structure.invoke({"chemical_symbol": "Al"})
        cls.structure = get_atom_dict_equilibrated_structure.invoke(
            {"atom_dict": atom_dict, "calculator_str": cls.calculator_str}
        )

    def test_structure(self):
        self.assertTrue(np.all(np.isclose(self.structure.positions, [[0.0, 0.0, 0.0]])))
        self.assertEqual(self.structure.numbers, [13])
        self.assertTrue(all(self.structure.pbc))
        self.assertTrue(np.isclose(np.linalg.det(self.structure.cell), 16.736861180699186, atol=10**-4))

    def test_bulk_modulus(self):
        b0 = get_bulk_modulus.invoke({"atom_dict": self.structure, "calculator_str": self.calculator_str})
        self.assertTrue(np.isclose(b0, 63.508867430434144, atol=10**-2))

    def test_equilibrium_volume(self):
        v0 = get_equilibrium_volume.invoke({"atom_dict": self.structure, "calculator_str": self.calculator_str})
        self.assertTrue(np.isclose(v0, 16.736756950577675, atol=10**-4))


class TestExperimentalReference(TestCase):
    def test_wikipedia(self):
        self.assertEqual(
            get_experimental_elastic_property_wikipedia.invoke({"chemical_symbol": "Al"}),
            {
                'bulk_modulus': 76.0,
                'poissons_ratio': 0.35,
                'shear_modulus': 26.0,
                'youngs_modulus': 70.0,
            },
        )

    def test_mendeleev(self):
        self.assertEqual(get_chemical_information_from_mendeleev.invoke("Al")["atomic_number"], 13)

