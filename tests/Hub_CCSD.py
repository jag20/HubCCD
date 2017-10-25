import sys
sys.path.append('../')
import numpy as np
import CCSD
import ham

#This script does UCCSD on a small doped hubbard lattice.

flnm = "./Output"
print("Writing results to ", flnm)
with open(flnm, "w") as f:
   f.write("U" + "     " + "Ecorr" + "            "+ "Escf" + "\n")
   U = 2.0
   print("U = ", U)
   #Build hamiltonian and do RHF
   hub = ham.hub(nsitesx=4,nsitesy=2,U=U,fill=3./8.,PeriodicX=False)
   hub.get_ints(wfn_type="uhf")
   #Do attenuated CCSD
   CCSD.ccsd(hub)
   print("ECCSD = ", hub.ecorr, "escf = ", hub.escf)
   f.write(str(U) + "   " + str(hub.ecorr) + "  "+ str(hub.escf) + "\n")
 

