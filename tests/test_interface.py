from unittest import TestCase
import numpy as np
from llm_compmat.tools.interface import (
    get_equilibrium_lattice,
    get_bulk_modulus,
    get_equilibrium_volume,
    get_experimental_elastic_property_wikipedia,
    get_element_property_mendeleev,
)


class TestInterfaceEMT(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculator_str = "emt"
        cls.structure = get_equilibrium_lattice.invoke({"chemical_symbol": "Al", "calculator_str": cls.calculator_str})

    def test_structure(self):
        self.assertTrue(np.all(np.isclose(self.structure.positions, [[0.0, 0.0, 0.0]])))
        self.assertEqual(self.structure.numbers, [13])
        self.assertTrue(all(self.structure.pbc))
        self.assertAlmostEqual(np.linalg.det(self.structure.cell), 15.931389072530033)

    def test_bulk_modulus(self):
        b0 = get_bulk_modulus.invoke({"atom_dict": self.structure, "calculator_str": self.calculator_str})
        self.assertAlmostEqual(b0, 39.518515298270835)

    def test_equilibrium_volume(self):
        v0 = get_equilibrium_volume.invoke({"atom_dict": self.structure, "calculator_str": self.calculator_str})
        self.assertAlmostEqual(v0, 15.931553356465589)


class TestInterfaceMace(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculator_str = "mace"
        cls.structure = get_equilibrium_lattice.invoke({"chemical_symbol": "Al", "calculator_str": cls.calculator_str})

    def test_structure(self):
        self.assertTrue(np.all(np.isclose(self.structure.positions, [[0.0, 0.0, 0.0]])))
        self.assertEqual(self.structure.numbers, [13])
        self.assertTrue(all(self.structure.pbc))
        self.assertAlmostEqual(np.linalg.det(self.structure.cell), 16.736861180699186)

    def test_bulk_modulus(self):
        b0 = get_bulk_modulus.invoke({"atom_dict": self.structure, "calculator_str": self.calculator_str})
        self.assertAlmostEqual(b0, 63.866120321206715)

    def test_equilibrium_volume(self):
        v0 = get_equilibrium_volume.invoke({"atom_dict": self.structure, "calculator_str": self.calculator_str})
        self.assertAlmostEqual(v0, 16.736756950577675)


class TestExperimentalReference(TestCase):
    def test_wikipedia(self):
        self.assertEqual(
            get_experimental_elastic_property_wikipedia.invoke({"chemical_symbol": "Al", "property": "youngs_modulus"}),
            "70 GPa")
        self.assertEqual(
            get_experimental_elastic_property_wikipedia.invoke({"chemical_symbol": "Al", "property": "poissons_ratio"}),
            "0.35")
        self.assertEqual(
            get_experimental_elastic_property_wikipedia.invoke({"chemical_symbol": "Al", "property": "bulk_modulus"}),
            "76 GPa")
        self.assertEqual(
            get_experimental_elastic_property_wikipedia.invoke({"chemical_symbol": "Al", "property": "shear_modulus"}),
            "26 GPa")

    def test_mendeleev(self):
        self.assertEqual(get_element_property_mendeleev.invoke({"chemical_symbol": "Al", "property": "atomic_number"}), "13")

