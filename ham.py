import numpy as np
from scf import *
#from readints import read_ints 
#This module builds the Hamiltonian object needed for post-Hartree-Fock methods (i.e. MO-basis integrals). 
#Current support is restricted to molecules (which require input of external integral file, RHF only for now) and
#1D and 2D Hubbard with nearest-neighbor interaction (open or periodic boundary conditions).
#For hub class builds the on-site basis integrals and performs RHF or UHF as requested by user, and then transforms the AO integrals
#into the MO basis.

class ham():

  def __init__(self):
    #We used to have more things here before rewriting CCD code, keep just in case.
    self.hamtype = "Hamiltonian"

class hub(ham):

  def __init__(self,U=1,t=1,nsitesx=6,nsitesy=1,fill=0.5,PeriodicX=True,PeriodicY=False):
#    super(hub, self).__init__()
    ham.__init__(self)
    self.hamtype = "Hubbard"
    self.nbas = nsitesx*nsitesy
    self.nsitesx = nsitesx
    self.nsitesy = nsitesy
    self.nocc = int(self.nbas*fill)
    self.U = U
    self.hubT = -t
    self.nvirt = self.nbas-self.nocc
    self.PeriodicX = PeriodicX
    self.PeriodicY = PeriodicY


  def get_ints(self,wfn_type="none",denfile="none",guess="af"):
   #Do AO integrals and RHF by default
   self.wfn_type = wfn_type.lower()
#   print(self.wfn_type)
   #Two-electron integrals
   self.Eri =  np.zeros((self.nbas,self.nbas,self.nbas,self.nbas))
   np.fill_diagonal(self.Eri,self.U)
   #One-electron integrals
   self.OneH= np.zeros((self.nbas,self.nbas))
   for ix in range(self.nsitesx):
     for iy in range(self.nsitesy):
       i = ix + (iy)*self.nsitesx
       #Set index j of site to interact with

       #Interact with neighbors in the x direction
       if (ix != (self.nsitesx -1)):
         #Interact with neighbor to the right
         j = (ix+1) + (iy)*self.nsitesx
         self.OneH[i,j] = self.hubT
         self.OneH[j,i] = self.hubT
       else:
         #interact with the left-most site if we're at right edge of the lattice and have PBC
         if self.PeriodicX:
           j = (iy)*self.nsitesx
           self.OneH[i,j] = self.hubT
           self.OneH[j,i] = self.hubT

       #Interact with neighbors in the y direction
       if (iy != (self.nsitesy -1)):
         #interact with neighbor above
         j = ix + (iy+1)*self.nsitesx
         self.OneH[i,j] = self.hubT
         self.OneH[j,i] = self.hubT
       else:
         #interact with the bottom site if we have PBC
         if self.PeriodicY:
           j = ix
           self.OneH[i,j] = self.hubT
           self.OneH[j,i] = self.hubT

#   if self.PeriodicX:
##     print("Periodic in X")
#     self.OneH[0,-1] = self.hubT
#     self.OneH[-1,0] = self.hubT

##  We can just use the eigenvectors of the core Hamiltonian as the RHF basis for Hubbard.
#   if (wfn_type == "rhf"):
#     #get scf-eigenvectors
#     e, C = np.linalg.eigh(self.OneH) #very important to have orthogonal MOs, so use eigh instead of eig
#     idx = e.argsort()
#     e = e[idx]
#     C = C[:,idx]
#  
#     #Build Density and Fock matrices (Szabo and Ostlund, Modern Quantum Chemistry)
#     P = buildP(C,self.nocc)
#     self.F = buildF(P,self.OneH,self.Eri)
#     self.escf = calc_escf(P,self.OneH,self.F)
#     self.F, self.Eri = MO_tran(self.F,self.Eri,C)
#
   if (wfn_type == "rhf"):
   #do RHF and transform to MO basis for Post-HF calculation
     RHF(self,denfile)
     self.F   = onee_MO_tran(self.F,self.C)
     self.Eri = twoe_MO_tran(self.Eri,self.C,self.C)

   elif (wfn_type == "uhf"):
   #Do UHF and transform to MO basis for Post-HF calculation. We now have support for spin-summed UCCSD and UCCD coupled cluster. No n   need to transform to spin-orbital basis if just doing UCC.
     self.F_a, self.F_b, self.C_a, self.C_b = UHF(self,denfile,guess)
     self.nocca = self.nocc
     self.noccb = self.nocc
     self.nvirta = self.nvirt
     self.nvirtb = self.nvirt

     self.F_a = onee_MO_tran(self.F_a,self.C_a)
     self.F_b = onee_MO_tran(self.F_b,self.C_b)
     self.Eri_aa = twoe_MO_tran(self.Eri,self.C_a,self.C_a)
     self.Eri_ab = twoe_MO_tran(self.Eri,self.C_a,self.C_b)
     self.Eri_bb = twoe_MO_tran(self.Eri,self.C_b,self.C_b)

	#Do the antisymmetrization inside the CC routines to avoid having to do/undo/redo it depending on combination of scf and cc
#     self.Eri_aa = self.Eri_aa - np.swapaxes(self.Eri_aa,2,3)  #antisymmetrize
#     self.Eri_bb = self.Eri_bb - np.swapaxes(self.Eri_bb,2,3)  #antisymmetrize
#
   elif (wfn_type == "ghf"):
   #Our spin-orbital CCD and CCSD codes could do true GHF-CCD, but do not have support for finding actual Sz-broken GHF solutions 		 right now, so we just convert uhf integrals to spin-orbital basis.
     F_a, F_b, C_a, C_b = UHF(self,denfile,guess)
     F_GHF, Eri_GHF, C_GHF = ao_to_GHF(C_a,C_b,F_a,F_b,self.Eri,self.nocc,self.nocc,self.nbas)
     self.F, self.Eri, self.C = np.copy(F_GHF), np.copy(Eri_GHF), np.copy(C_GHF)
     self.nbas = 2*self.nbas
     self.nocc = 2*self.nocc
     self.nvirt = 2*self.nvirt

class mol(ham):

  def __init__(self,wfn_type,fname,denfile="none"):
    from readints import read_ints 
    hub.__init__(self)
    self.hamtype = "Molecule"
    self.wfn_type = wfn_type.lower()
    #We can only do molecular calculations if we read integrals from a Gaussian16 Matrix Element File for now.
    if (self.wfn_type == 'rhf'):
      self.nbas, self.nocc, self.nvirt, self.escf, self.C, self.F, self.Eri = read_ints(self.wfn_type,fname)
    elif (self.wfn_type == 'uhf'):
       self.nbas, self.nocca, self.noccb, self.nvirta, self.nvirtb, self.escf, self.C_a, self.C_b, self.F_a, self.F_b, self.Eri_aa, self.Eri_ab, self.Eri_bb = read_ints(self.wfn_type,fname)
    elif (self.wfn_type == 'rohf'): 
       self.nbas, self.nocca, self.noccb, self.nvirta, self.nvirtb, self.escf, self.nrep, C_a, C_b, P_a, P_b, X, S, OneH, Eri = read_ints(self.wfn_type,fname)
	   #Take P to Orthogonal basis
#       self.P_a = np.dot(np.linalg.inv(X),np.dot(P_a,np.linalg.inv(X).T)) 
#       self.P_b = np.dot(np.linalg.inv(X),np.dot(P_b,np.linalg.inv(X).T)) 
#       #Take integrals back to Orthogonal Basis
#       MOa_to_Orth = np.dot(np.linalg.inv(self.C_a),X)
#       self.OneH = onee_MO_tran(H,MOa_to_Orth)

#	   #Keep Eris in Mulliken notation for scf
#       self.Eri = np.swapaxes(self.Eri_aa,1,2)
#       self.Eri = np.swapaxes(twoe_MO_tran(self.Eri,MOa_to_Orth,MOa_to_Orth),1,2)
#       self.Eri = np.swapaxes(twoe_MO_tran(self.Eri,self.C_a,self.C_b),1,2)

## test
#       print("TEST = ", calc_euhf(self.P_a,self.P_b,F_a,F_b,self.OneH) + self.nrep)
       #1-es back to AO basis
#       MOa_to_AO = np.linalg.inv(C_a)
#       F_a = onee_MO_tran(F_a,MOa_to_AO)
#       F_b = onee_MO_tran(F_b,MOa_to_AO)
#       self.Eri = twoe_MO_tran(self.Eri_aa,MOa_to_AO,MOa_to_AO) 

       F_a, F_b = buildFs_uhf(P_a,P_b,OneH,Eri)

       etest = (calc_euhf(P_a,P_b,F_a,F_b,OneH) + self.nrep)
       if abs(etest  - self.escf > 1.0e-7):
         print("Error recovering integrals")
       #take everything to AO-basis
       P_a = np.dot(np.linalg.inv(X),np.dot(P_a,np.linalg.inv(X).T)) 
       P_b = np.dot(np.linalg.inv(X),np.dot(P_b,np.linalg.inv(X).T)) 
       OneH = onee_MO_tran(OneH,X)
       F_a = onee_MO_tran(F_a,X)
       F_b = onee_MO_tran(F_b,X)
       Eri = np.swapaxes(twoe_MO_tran(Eri,X,X),1,2)

       #Build charge density matrix and get transformation to NO basis
       P = 0.5e0*(P_a+P_b)
       occ_nums, AO_to_NO = np.linalg.eigh(P)
       idx = (-occ_nums).argsort()
       occ_nums = occ_nums[idx]
       AO_to_NO = AO_to_NO[:,idx]
	   #Transform integrals to NO basis
       self.F_a = onee_MO_tran(F_a,AO_to_NO)
       self.F_b = onee_MO_tran(F_b,AO_to_NO)
       self.Eri_aa = twoe_MO_tran(Eri,AO_to_NO,AO_to_NO) 
       self.Eri_ab = np.copy(self.Eri_aa)
       self.Eri_bb = np.copy(self.Eri_aa)

       self.wfn_type = 'uhf' #post-HF routines don't care about S^2 constraints on MOs
#       stop


#    elif (self.wfn_type == 'ghf'):
#       print('shape =', self.Eri.shape)
#       
#       F_GHF, Eri_GHF, C_GHF = moUHF_to_GHF(self.C_a,self.C_b,self.F_a,self.F_b,self.Eri_aa,self.nocca,self.noccb,self.nbas)
# 
#       self.F, self.Eri, self.C = np.copy(F_GHF), np.copy(Eri_GHF), np.copy(C_GHF)
#       self.nbas = 2*self.nbas
#       self.nocc = self.nocca + self.noccb
#       self.nvirt = self.nvirta + self.nvirtb
#       print('shape =', self.Eri.shape)
#
