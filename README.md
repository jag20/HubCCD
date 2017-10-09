# HubCCD
This set of python modules implements RHF and several CCD-based post-RHF methods: CCD, ACPQ, Linearized CCD, and AttCCD developed by a number of groups. We also have UHF and UHF-based CCD (UCCD).
The SCF and Post-SCF routines can be used for general Hamiltonians, but the code only knows
how to generate integrals for 1D and 2D Hubbard lattices with nearest-neighbor interactions
and open or periodic boundary conditions.

# Motivation
This code base is useful for benchmarking and rapid implementation of CCD-based theories.


# Usage
Usage of particular functions is defined in the code. A sample excution script 'run.py' 
demonstrates how to run AttCCD on the 10-site 1D Hubbard Hamiltonian at Half-filling with
periodic boundary conditions.
