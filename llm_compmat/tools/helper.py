from ase.calculators.emt import EMT
from mace.calculators import mace_mp


def get_calculator(calculator_str: str = "emt"):
    if calculator_str == "emt":
        return EMT()
    elif calculator_str == "mace":
        return mace_mp(
            model="medium",
            dispersion=False,
            default_dtype="float32",
            device='cpu'
        )