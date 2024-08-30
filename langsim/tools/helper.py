

def get_calculator(calculator_str: str = "emt"):
    if calculator_str == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    elif calculator_str == "mace":
        from mace.calculators import mace_mp
        return mace_mp(
            model="medium", dispersion=False, default_dtype="float32", device="cpu"
        )
