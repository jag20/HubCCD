import sys
sys.path.append('../')
import ham 
import numpy as np
import CCD

#This script demonstrates the use of the CCD and CCDutils modules
#to run attenuated Coupled Cluster(Gomez, Henderson and Scuseria, Mol. Phys. onccde 23 Mar 2017)
#for N2 at 3 Bohr bondlength in the STO-3G basis, reading from the Gaussian Matrix Element file N2.mat

ml = ham.mol('rhf','N2.mat')
CCD.ccd(ml,variant='att')
print("E(AttCCD) = ", ml.ecorr, "escf = ", ml.escf)
