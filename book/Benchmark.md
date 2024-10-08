# Benchmark
The hallucinations of large language models (LLM) limit their application to control atomistic simulation. Especially
it is hard for beginners to judge the results predicted from the LLM. Even when very specific inputs are provided, the
current generation of LLMs struggle to create the input for popular software tools like the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

Example Input:
```
Calculate the bulk modulus for aluminium with ASE using the EMT simulation code.
```

## ChatGPT 3.5
On the May 2, 2024 replied with the following Python code: 
```python
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import write
import numpy as np

# Step 1: Create aluminum structure
aluminum = Atoms('Al', [(0, 0, 0)], cell=(5, 5, 5))

# Step 2: Relax the structure
relaxation = BFGS(aluminum)
relaxation.run(fmax=0.01)

# Step 3: Apply hydrostatic pressure
pressures = np.linspace(0, 10, 11)  # Example pressures in GPa
energies = []

for pressure in pressures:
    volume_scaling_factor = (1 - 3 * pressure * 1e-9) ** (1 / 3)
    strained_structure = aluminum.copy()
    strained_structure.set_cell(aluminum.cell * volume_scaling_factor, scale_atoms=True)
    
    # Step 4: Calculate energy
    strained_structure.set_calculator(EMT())
    energy = strained_structure.get_potential_energy()
    energies.append(energy)

# Step 5: Fit equation of state
# (Not included here, but you can use scipy.optimize.curve_fit)

# Step 6: Determine bulk modulus
# (From the fitted equation of state)
```
This is wrong in a number of ways: 
* No code is provided for Step 5 and Step 6 - a beginner user without prior experience with ASE has no chance to 
  complete this Python script. 
* By applying 11 pressures ranging from 0 to 10, only the right half of the energy volume curve is calculated, so even
  with the input to steps 5 and 6 this script would lead to poor predictions. 
* Finally, the script has unused imports like `from ase.io import write` and is not using the `calculate_eos()` function
  which the ASE provides to simplify energy volume curve calculation. 

## Chat GPT 4.0
On the May 2, 2024 replied with the following Python code: 
```python
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import calculate_eos

# Create a bulk aluminium structure (face-centered cubic)
al = bulk('Al', 'fcc', a=4.05)  # You can adjust 'a' to experimental lattice constant

# Set up the EMT calculator
al.calc = EMT()

# Calculate the equation of state
eos = calculate_eos(al)
v0, e0, B = eos.fit()

# Print the bulk modulus (B is in eV/Angstrom^3 which can be converted to GPa)
print(f"Bulk Modulus: {B / 1.60219e-19 * 1e21 / 1e9} GPa")
```
This code uses the `calculate_eos()` function ASE provides to simplify the calculation of the bulk modulus and it is a
complete code example which can be executed without any modification. The only mistake is the unit conversion to 
Gigapascal (GPa). The correct unit conversion is: 
```python
from ase.units import kJ
print(B / kJ * 1.0e24, 'GPa')
```
This information is even provided in the documentation of the `calculate_eos()` function:
```
Signature: eos.fit(warn=True)
Docstring:
Calculate volume, energy, and bulk modulus.

Returns the optimal volume, the minimum energy, and the bulk
modulus.  Notice that the ASE units for the bulk modulus is
eV/Angstrom^3 - to get the value in GPa, do this::

  v0, e0, B = eos.fit()
  print(B / kJ * 1.0e24, 'GPa')
```
So the result provided by Chat GPT 4.0 is 90% correct and a scientist without prior knowledge of ASE would be able to 
correct the unit conversion, but the risk of disregarding a calculation because of a wrong unit conversion is too high.

## ChatGPT o1-preview
On the September 19, 2024 replied with the following Python code: 
```
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState
import matplotlib.pyplot as plt

# Step 1: Set up the aluminum bulk structure
# Create an FCC aluminum unit cell with a lattice constant of 4.05 Å
atoms = bulk('Al', 'fcc', a=4.05)

# Step 2: Assign the EMT calculator
atoms.calc = EMT()

# Prepare arrays to store volumes and energies
volumes = []
energies = []

# Step 3: Compute energies at different volumes
# Define scaling factors around the equilibrium lattice constant
scaling_factors = np.linspace(0.94, 1.06, 13)  # 13 points from 0.94 to 1.06

for sf in scaling_factors:
    # Scale the unit cell
    scaled_atoms = atoms.copy()
    scaled_atoms.set_cell(atoms.cell * sf, scale_atoms=True)
    
    # Calculate the potential energy and volume
    energy = scaled_atoms.get_potential_energy()
    volume = scaled_atoms.get_volume()
    
    # Store the results
    energies.append(energy)
    volumes.append(volume)

# Step 4: Fit the energies and volumes to an equation of state
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()

# Step 5: Extract and print the bulk modulus
# Convert bulk modulus from eV/Å³ to GPa (1 eV/Å³ = 160.21766208 GPa)
bulk_modulus_gpa = B * 160.21766208

print(f"Equilibrium volume (V0): {v0:.2f} Å³")
print(f"Equilibrium energy (E0): {e0:.2f} eV")
print(f"Bulk modulus (B): {bulk_modulus_gpa:.2f} GPa")

# Optional: Plot the equation of state
eos.plot('aluminum_eos.png')
plt.show()
```
As you can see the suggested solution from `o1-preview` is again closer to the solution from ChatGPT 3.5 rather than 4.0.
While the unit conversion is now fixed, there is no calculator object set, so when you execute the code above you get
the following error message:
```
RuntimeError: Atoms object has no calculator.
```
So for this specific application ChatGPT 4.0 provides the best solution. 

## Summary
While the performance improves with increasing training size from ChatGPT 3.5 to 4.0, the risk of small hallucinations
like a wrong unit conversion leading to a wrong calculation result is too high. For science, it is not sufficient to be
right 90% of the time or even 99%. 

Based on this experience, the LangSim team decided to develop simulation agents which can be called from the LLM to 
produce reliable and scientifically correct predictions. For this purpose, the [Langchain](https://www.langchain.com/) 
framework was used as interface to LLMs, [ASE](https://wiki.fysik.dtu.dk/ase/) and [atomistics](https://atomistics.readthedocs.io)
were used to construct simulation workflows, [emt](https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html) and [mace](https://mace-docs.readthedocs.io/en/latest/)
as simulation engines and [mendeleev](https://mendeleev.readthedocs.io) and [wikipedia](https://en.wikipedia.org/wiki/Main_Page) 
for experimental references. 
