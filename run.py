import ham 
import numpy as np
import CCD

#This script demonstrates the use of the CCD and CCDutils modules
#to run attenuated Coupled Cluster(Gomez, Henderson and Scuseria, Mol. Phys. online 23 Mar 2017)
#for 1D 10-site Hubbard at half filling with periodic boundary conditions for a range of U values

flnm = "out"
with open(flnm, "w") as f:
   f.write("U" + "     " + "Ecorr" + "            "+ "Escf" + "\n")
   #for U in np.arange(1.,10.1,.2):
   for U in np.arange(10,1,-.2):
    print("U = ", U)
    #Build hamiltonian and do RHF
    hub = ham.hub(nsitesx=6,nsitesy=1,U=U,fill=0.5,PeriodicX=False)
#    hub = ham.hub(nsitesx=4,nsitesy=2,U=U,fill=3.0/8.0,PeriodicX=True)
    hub.get_ints(wfn_type="uhf",denfile="dense")
    CCD.ccd(hub,ampfile="amps",variant="ccd")
    f.write(str(U) + "   " + str(hub.eccd) + "  "+ str(hub.escf) + "\n")
#    f.write(str(U) + "   " + "  "+ str(hub.escf) + "\n")
 

