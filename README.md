# HubCCD
This set of python3 modules implements RHF and several CCD-based post-RHF methods: CCD, AttCCD, ACPQ, Linearized CCD and parameterized CCD, which have been developed by a number of groups.
Specific references are included in the source. We also have UHF and UHF-based CCD (UCCD). UCCD uses a spin-orbital basis CCD code that could employ a GHF
reference, but we have not implemented GHF yet.
The SCF and Post-SCF routines can be used for general Hamiltonians, but the code only knows
how to generate integrals for 1D and 2D Hubbard lattices with nearest-neighbor interactions
and open or periodic boundary conditions.

Required Packages: numpy, os, pickle. If molecular calculations are desired, we also need Gaussian16 and their Fortran/python
routines for accessing Gaussian matrix element files: http://gaussian.com/interfacing/

# Motivation
This code base is useful for benchmarking and rapid implementation of CCD-based theories.
The Hubbard Hamiltonian is a standard model of strong-correlation and is a useful model system
with interesting physics.


# Usage
Usage of particular functions is defined in the code. A sample excution script 'run.py' 
demonstrates how to run AttCCD on the 10-site 1D Hubbard Hamiltonian at half-filling with
periodic boundary conditions.

# Contributors
John Gomez, Rice University

# License
MIT License.
