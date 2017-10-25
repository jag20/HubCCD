import sys
sys.path.append('../')
import numpy as np
import CCD
import ham

#This script demonstrates the use of the CCD and CCDutils modules
#to run attenuated Coupled Cluster(Gomez, Henderson and Scuseria, Mol. Phys. online 23 Mar 2017)
#for 1D 10-site Hubbard at half filling with periodic boundary conditions for a range of U values

flnm = "./Output"
print("Writing results to ", flnm)
with open(flnm, "w") as f:
   f.write("U" + "     " + "Ecorr" + "            "+ "Escf" + "\n")
   for U in np.arange(1.,20.1,1):
    print("U = ", U)
    #Build hamiltonian and do RHF
    hub = ham.hub(nsitesx=10,nsitesy=1,U=U,fill=0.5,PeriodicX=True)
    hub.get_ints(wfn_type="rhf")
	#Do attenuated CCD
    CCD.ccd(hub,ampfile="none",variant="att")
    print("ECCD = ", hub.ecorr, "escf = ", hub.escf)
    f.write(str(U) + "   " + str(hub.ecorr) + "  "+ str(hub.escf) + "\n")
 

