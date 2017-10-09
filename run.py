import ham 
import numpy as np
import CCD

#This script demonstrates the use of the CCD and CCDutils modules
#to run attenuated Coupled Cluster(Gomez, Henderson and Scuseria, Mol. Phys. online 23 Mar 2017)
#for 1D 10-site Hubbard at half filling with periodic boundary conditions for a range of U values

flnm = "out"
with open(flnm, "w") as f:
   f.write("U" + "     " + "Ecorr" + "            "+ "Escf" + "\n")
   for U in np.arange(1.,10.1,.2):
    print("U = ", U)
    #Build hamiltonian and do RHF
#    hub = ham.hub(nsitesx=6,nsitesy=1,U=U,fill=0.5,PeriodicX=False)
    hub = ham.hub(nsitesx=4,nsitesy=2,U=U,fill=0.5,PeriodicX=False)
    hub.get_ints(wfn_type="uhf")
#    CCD.ccd(hub,ampfile="amps",variant="att")
#    f.write(str(U) + "   " + str(hub.eccd) + "  "+ str(hub.escf) + "\n")
    f.write(str(U) + "   " + "  "+ str(hub.escf) + "\n")
 

