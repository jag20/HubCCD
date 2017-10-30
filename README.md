# HubCCD
This set of python3 modules implements Hartree-Fock and several coupled-cluster based post-Hartree-Fock methods developed by a
number of groups, particularly for strong correlation. References are listed in the source. 

Methods:

Integrals:
-Integral generation for 1D and 2D Hubbard Hamiltonian with open or periodic boundary conditions with nearest-neighbor interactions.
-Interfacing with Gaussian16 matrix element files to obtain integrals for molecular calculations.
If molecular calculations are desired, we also need Gaussian16 and their Fortran/python
routines for accessing Gaussian matrix element files: http://gaussian.com/interfacing/

Hartree-Fock:
-Restricted Hartree-Fock (molecules or Hubbard. See scf.py)
-Unrestricted Hartree-Fock (molecules or Hubbard, but S^2-breaking guesses only available for Hubbard. We typically read converged
HF solutions for molecules from an external program. See scf.py)


Spin-summed RHF-based coupled cluster and variants (See CCD.py and CCDutils.py):
-Coupled cluster with doubles (CCD)
-Linearized CCD
-Attenuated CCD (AttCCD)
-Approximate coupled pair theory with quadruples (ACPQ)
-Parametrized CCD

Spin-orbital coupled cluster and variants (See CCSD.py and CCSDutils.py):
-CCD and CCSD that can use RHF, UHF or GHF reference
-spin-orbital variants of several of the above methods.


Spin-Summed UHF-based coupled cluster and variants (See UCCSD.py, UCCSDG1.f90 and UCCSDG2.f90)
-UCCSD
***Note: The tensor contractions are done in the fortran modules UCCSDG1.f90 and UCCSDG2.f90, these must be compiled into
python modules using f2py:  
f2py -c -m UCCSDG2 UCCSDG2.py
f2py -c -m UCCSDG1 UCCSDG1.py
Or see "Makefile", and compile using the command "make all."



Required Packages: numpy, os, pickle, f2py.
If molecular calculations are desired, we also need Gaussian16 and their Fortran/python
routines for accessing Gaussian matrix element files: http://gaussian.com/interfacing/

# Motivation
This code base is useful for benchmarking and rapid implementation of CCD-based theories.
The Hubbard Hamiltonian is a standard model of strong-correlation and is a useful model system
with interesting physics.


# Usage
Usage of particular functions is defined in the code. 
A couple of sample scripts are included in ./tests

# Contributors
John Gomez, Rice University

# License
MIT License.
