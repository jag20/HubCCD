import numpy as np
from scf import *
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
    super(hub, self).__init__()
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


  def get_ints(self,wfn_type="rhf",denfile="none"):
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
       if (ix != (self.nsitesx -1)):
         #Interact with neighbors in the x-direction
         j = (ix+1) + (iy)*self.nsitesx
         self.OneH[i,j] = self.hubT
         self.OneH[j,i] = self.hubT
       else:
         #interact with the left-most site if we're at right edge of the lattice and have PBC
         if self.PeriodicX:
           j = 0
           self.OneH[i,j] = self.hubT
           self.OneH[j,i] = self.hubT

       #Interact with neighbors in the y-direction
       if (iy != (self.nsitesy -1)):
         j = ix + (iy+1)*self.nsitesx
         self.OneH[i,j] = self.hubT
         self.OneH[j,i] = self.hubT
       else:
         #interact with the bottom site if we have PBC
         if self.PeriodicY:
           j = 0
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
     self.F, self.Eri = MO_tran(self.F,self.Eri,self.C)

   elif (wfn_type == "uhf"):
   #Do UHF and transform to MO basis for Post-HF calculation. Note we only have support for spin-orbital coupled cluster at this point. 
   #We could do true GHF-CCD, but do not have support for finding actual Sz-broken GHF solutions right now.
     F_a, F_b, C_a, C_b = UHF(self,denfile)
     F_GHF, Eri_GHF, C_GHF = UHF_to_GHF(C_a,C_b,F_a,F_b,self.Eri,self.nocc,self.nocc,self.nbas)
     self.F, self.Eri, self.C = np.copy(F_GHF), np.copy(Eri_GHF), np.copy(C_GHF)
     self.nbas = 2*self.nbas
     self.nocc = 2*self.nocc
     self.nvirt = 2*self.nvirt

class mol(ham):

  def __init__(self):
    super(mol, self).__init__()
    self.hamtype = "Molecule"

  def get_ints(self,fname,wfn_type="rhf"):
    self.wfn_type = wfn_type
    self.intfile=fname
    with open(fname) as f:
      fle = f.readlines()
      self.nbas = int(fle[0])
      self.nocc = int(fle[1])
      self.nvirt = self.nbas-self.nocc 
      self.escf = float(fle[-1])
      self.Eri = np.array(fle[2:self.nbas**4+2]).reshape((self.nbas,self.nbas,self.nbas,self.nbas), order='F')
      self.F = np.array(fle[self.nbas**4+2: self.nbas**4+2 + self.nbas**2]).reshape([self.nbas,self.nbas], order = 'F')
      self.Eri = self.Eri.astype(float)
      self.F= self.F.astype(float)

