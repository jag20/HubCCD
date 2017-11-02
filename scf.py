import numpy as np
import os
import pickle 
#from anaden import *
#This module implements RHF and UHF for arbitrary Hamiltonians. We only need AO-basis 1 and 2-electron integrals, although the code
#does assume the basis is orthonormal. These algorithms are taken exactly from Szabo and Ostlund, Modern Quantum Chemistry.

def RHF(ham,denfile):
  #guess core Hamiltonian for Fock matrix guess. For Hubbard, we get plane-wave as the RHF basis (eigenvectors of the 
  #one-electron integrals
  print("Doing RHF")
  P = np.zeros([ham.nbas,ham.nbas])
  ham.F = buildF(P,ham.OneH,ham.Eri)
  error = 1.0
  tol = 1.0e-12
  eold = 1.0
  niter = 1
  while (error  > tol):
    e, C = np.linalg.eigh(ham.F) 
    idx = e.argsort()
    e = e[idx]
    ham.C = C[:,idx]
    P = buildP(C,ham.nocc)
    ham.F = buildF(P,ham.OneH,ham.Eri)
    ham.escf = calc_escf(P,ham.OneH,ham.F)
#    print( "iteration = ", niter, "escf = ", ham.escf)
    error = abs(eold-ham.escf)
    eold = ham.escf
    niter += 1


def UHF(ham,denfile="none",guess="af"):
  #Do UHF. Anti-ferromagnetic guess used here is only useful for Hubbard, but the UHF algorithm itself 
  #as implemented here can be used
  #for arbitrary Hamiltonians.

  print("Doing UHF")

  #Add a check for Fortran-ordered, text-based alpha and beta MO coefficients 
  #to interface with Ethan's code
  fla="cA"
  flb="cB"
  if ( os.path.isfile(fla) and os.path.isfile(flb)):
    print("Using UpCCD basis")
    with open(fla, 'r') as f:
      C_a = []
      stuff = f.readlines()
      for line in stuff:
        for item in line.split():
#          print("%.16f" %float(item))
          C_a = np.append(C_a,float(item))

    with open(flb, 'r') as f:
      C_b = []
      stuff = f.readlines()
      for line in stuff:
        for item in line.split():
#          print("%.16f" %float(item))
          C_b = np.append(C_b,float(item))

    C_a = C_a.reshape((ham.nbas,ham.nbas),order = 'F')
    C_b = C_b.reshape((ham.nbas,ham.nbas),order = 'F')
    P_a = (buildP(C_a,ham.nocc))
    P_b = (buildP(C_b,ham.nocc))
    F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
    ham.escf = calc_euhf(P_a,P_b,F_a,F_b,ham.OneH)
    ham.Pa = np.copy(P_a)
    ham.Pb = np.copy(P_b)

    return F_a, F_b, C_a, C_b


  #Read density matrix from file if we have it
  if ((denfile.lower() != "none") and os.path.isfile(denfile)):
    print("Reading previous density matrix")
    with open(denfile, 'rb') as f:
      P_a = pickle.load(f)
      P_b = pickle.load(f)
  #Otherwise, guess density matrix.
  #Break S^2 symmetry using an antiferromagnetic guess.
  #Make nsitesy copies of an nsitesx-dimensional 1-D lattice.
  else:
      guess = guess.lower()
      if (guess == "af"):
        print("Antiferromagnetic guess for UHF")
#        nmax = int(ham.nocc/ham.nsitesy) + 1
        nmax = ham.nsitesx
        den  = 2*ham.nocc/(ham.nsitesx*ham.nsitesy)
        print("den = ", den)
        xlat = np.zeros(ham.nsitesx)
        for n in range(0,nmax,2):
#            xlat[n] = 1
            xlat[n] = den
        diag_a = []
        diag_b = []
        for i in range(ham.nsitesy):
          if i%2 == 0:
              diag_a = np.append(diag_a,xlat)
              diag_b = np.append(diag_b,xlat[::-1])
          else:
              diag_a = np.append(diag_a,xlat[::-1])
              diag_b = np.append(diag_b,xlat)
        P_a = np.diag(diag_a)
        P_b = np.diag(diag_b)
#        print(P_a)
#        print(P_b)

#        plt_spin_den(P_a,P_b,ham.nsitesx,ham.nsitesy)
      elif (guess == "sdw"):
        print("Spin-density wave guess for UHF")
        nmax = ham.nsitesx
        den  = 2*ham.nocc/(ham.nsitesx*ham.nsitesy)
        print("den = ", den)
        zeros = np.zeros(ham.nsitesx)
        xlat = np.ones(ham.nsitesx)*den
        diag_a = []
        diag_b = []
        for i in range(ham.nsitesy):
          if i%2 == 0:
              diag_a = np.append(diag_a,xlat)
              diag_b = np.append(diag_b,zeros)
          else:
              diag_b = np.append(diag_b,xlat)
              diag_a = np.append(diag_a,zeros)
        P_a = np.diag(diag_a)
        P_b = np.diag(diag_b)
        print(P_a)
        print(P_b)
#        F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
#        e, C_a = np.linalg.eigh(F_a)
#        idx = e.argsort()
#        e = e[idx]
#        C_a = C_a[:,idx]
#        e, C_b = np.linalg.eigh(F_b)
#        idx = e.argsort()
#        e = e[idx]
#        C_b = C_b[:,idx]
#        P_a = (buildP(C_a,ham.nocc))
#        P_b = (buildP(C_b,ham.nocc))
#        F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
#        ham.Pa = np.copy(P_a)
#        ham.Pb = np.copy(P_b)
#        escf = calc_euhf(P_a,P_b,F_a,F_b,ham.OneH)
#        ham.escf = escf
#
#        return F_a, F_b, C_a, C_b
#        stop

#        plt_spin_den(P_a,P_b,ham.nsitesx,ham.nsitesy)

  F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
  #Set some values and do the UHF iteration
  error = 1.0
  tol = 1.0e-12
  eold = 1.0
  niter = 1
  x = 5.0
  while (error  > tol):
    e, C_a = np.linalg.eigh(F_a)
    idx = e.argsort()
    e = e[idx]
    C_a = C_a[:,idx]
    e, C_b = np.linalg.eigh(F_b)
    idx = e.argsort()
    e = e[idx]
    C_b = C_b[:,idx]
    TempA = (buildP(C_a,ham.nocc))
    TempB = (buildP(C_b,ham.nocc))
    P_a = (TempA)/x + (x-1.0)/x*P_a  
    P_b = (TempB)/x + (x-1.0)/x*P_b
    F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
    escf = calc_euhf(P_a,P_b,F_a,F_b,ham.OneH)
#    print( "iteration = ", niter, "escf = ", escf)
    error = abs(eold-escf)
    eold = escf
    niter += 1
  
  ham.escf = escf

  if (denfile.lower() != "none"):
      with open(denfile, "wb") as f:
        pickle.dump(P_a, f)
        pickle.dump(P_b, f)

  ham.Pa = np.copy(P_a)
  ham.Pb = np.copy(P_b)
  return F_a, F_b, C_a, C_b



def ROHF(ham,denfile="none"):
  #This function implements ROHF as described in Tsuchimochi and Scuseria J. Chem. Phys. 133, 141102 (2010).
  #The algorithm is essentially UHF, with small modifications to the Fock matrix that come from Lagrange multipliers
  #That constrain the spin contamination to be zero.
  #In my work, I only have use for ROHF on molecules, so we will eliminate Hubbard-specific guesses that appear in UHF above.

  print("Doing ROHF")

  P_a = np.copy(ham.P_a)
  P_b = np.copy(ham.P_b)

#  F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
  #Set some values and do the UHF iteration
  error = 1.0
  tol = 1.0e-15
  eold = 1.0
  niter = 1
  x = 5.0
  while (error  > tol):

    F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
    escf = calc_euhf(P_a,P_b,F_a,F_b,ham.OneH) + ham.nrep
    #-----CUHF modification
    #Build charge density matrix and get transformation to NO basis
    P = 0.5e0*(P_a+P_b)
    occ_nums, AO_to_NO = np.linalg.eigh(P)
    idx = (-occ_nums).argsort()
    occ_nums = occ_nums[idx]
    AO_to_NO = AO_to_NO[:,idx]

    #Transform Fock matrices to NO basis
    F_a = onee_MO_tran(F_a,AO_to_NO)
    F_b = onee_MO_tran(F_b,AO_to_NO)

    #Modify the cv and vc terms a la CUHF (see reference). That is, set the core-virtual
    #blocks of the Fock matrices to be their closed-shell pieces
    F_core = 0.50e0*(F_a[:ham.noccb,ham.nocca:] + F_b[:ham.noccb,ham.nocca:])
    F_a[:ham.noccb,ham.nocca:] = F_core
    F_a[ham.nocca:,:ham.noccb] = F_core.T
    F_b[:ham.noccb,ham.nocca:] = F_core
    F_b[ham.nocca:,:ham.noccb] = F_core.T
   
    
    #Transform Fock matrices back to AO basis and continue with UHF iteration.
    F_a = onee_MO_tran(F_a,AO_to_NO.T)
    F_b = onee_MO_tran(F_b,AO_to_NO.T)
    #-----CUHF modification

#    #Get MO Coefficients
    e, C_a = np.linalg.eigh(F_a)
    idx = e.argsort()
    e = e[idx]
    C_a = C_a[:,idx]
    e, C_b = np.linalg.eigh(F_b)
    idx = e.argsort()
    e = e[idx]
    C_b = C_b[:,idx]
    #Build new Density Matrices, with level shift for convergence
    TempA = (buildP(C_a,ham.nocca))
    TempB = (buildP(C_b,ham.noccb))
    P_a = (TempA)/x + (x-1.0)/x*P_a  
    P_b = (TempB)/x + (x-1.0)/x*P_b
#    #Build final fock matrices
#    F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
    #check convergence on energy
    escf = calc_euhf(P_a,P_b,F_a,F_b,ham.OneH) + ham.nrep
    print( "iteration = ", niter, "escf = ", escf)
    error = abs(eold-escf)
    eold = escf
    niter += 1
 

  #Build final fock matrices
  F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
  ham.escf = escf + ham.nrep

  if (denfile.lower() != "none"):
      with open(denfile, "wb") as f:
        pickle.dump(P_a, f)
        pickle.dump(P_b, f)

  ham.Pa = np.copy(P_a)
  ham.Pb = np.copy(P_b)
  return F_a, F_b, C_a, C_b






def ao_to_GHF(C_a,C_b,F_a,F_b,Eriao,nocca,noccb,nbas):
  #Transform the UHF integrals to GHF form and transform to MO basis
  nvirta = nbas-nocca
  nvirtb = nbas-noccb
  C = np.zeros([nbas*2,nbas*2])
  F = np.zeros([nbas*2,nbas*2])
  Eri = np.zeros([nbas*2,nbas*2,nbas*2,nbas*2])
  #Build spin-orbital Mo coefficients
  C[:nbas,:nocca] = C_a[:,:nocca]
  C[nbas:,nocca:(nocca+noccb)] = C_b[:,:noccb]
  C[:nbas,(nocca+noccb):(nocca+noccb+nvirta)]  = C_a[:,nocca:]
  C[nbas:,(nocca+noccb+nvirta):] = C_b[:,noccb:]
   
  #build spinorbital fock
  F[:nbas,:nbas] = F_a
  F[nbas:,nbas:] = F_b
  F = onee_MO_tran(F,C)
 
  #Build Spinorbital 2-e integrals and transform to MO basis
  Eri[:nbas,:nbas,:nbas,:nbas] = Eriao
  Eri[:nbas,:nbas,nbas:,nbas:] = Eriao
  Eri[nbas:,nbas:,:nbas,:nbas] = Eriao
  Eri[nbas:,nbas:,nbas:,nbas:] = Eriao
#  Eri = twoe_MO_tran(Eri,C,C)
  Eri = twoe_MO_tran_UHF_to_GHF(Eri,C,C,nocca,noccb,nbas)
  Eri = Eri - np.swapaxes(Eri,2,3)  #antisymmetrize

  return F, Eri, C

def moUHF_to_GHF(C_a,C_b,F_a,F_b,Eri_aa,nocca,noccb,nbas):
  #We already have the MO integrals, but it's easier to get the ordering in the spin-orbital basis correct if we first
  #transform back to the AO basis and then multiply by the spin-orbital basis eigenvectors. 

  #Eri_aa to ao basis. The transformation also takes them back to back to mulliken order
  Eriao = twoe_MO_tran(Eri_aa,np.linalg.inv(C_a),np.linalg.inv(C_a))
  Fa_ao   = onee_MO_tran(F_a,np.linalg.inv(C_a))
  Fb_ao   = onee_MO_tran(F_b,np.linalg.inv(C_b))


  F, Eri, C = ao_to_GHF(C_a,C_b,Fa_ao,Fb_ao,Eriao,nocca,noccb,nbas)

  return F, Eri, C



#RHF utilities
def buildP(C,nocc):
  P = np.dot(C[:,:nocc],C.T[:nocc,:])
  return P

def buildF(P,OneH,Eri):
  F = np.copy(OneH)
  Gtemp = 2.0*Eri-np.swapaxes(Eri,1,3)
  F += np.einsum('ls,uvsl->uv',P,Gtemp)
  return F

def calc_escf(P,OneH,F):
  return np.einsum('vu,uv',P,(OneH+F))


#UHF Utilities
def buildFs_uhf(P_a,P_b,OneH,Eri):
  F_a = np.copy(OneH)
  F_a += np.einsum('ls,uvsl->uv',P_a+P_b,Eri)
  F_a -= np.einsum('ls,ulsv->uv',P_a,Eri)
 
  F_b = np.copy(OneH)
  F_b += np.einsum('ls,uvsl->uv',P_a+P_b,Eri)
  F_b -= np.einsum('ls,ulsv->uv',P_b,Eri)
  return F_a, F_b


def calc_euhf(P_a,P_b,F_a,F_b,OneH):
  escf =  np.einsum('vu,uv',(P_a + P_b), OneH)
  escf += np.einsum('vu,uv',(P_a), F_a)
  escf += np.einsum('vu,uv',(P_b), F_b)
  return escf/2.0




#Integral Transformation Utilities
def onee_MO_tran(F,C):
  F = np.dot(np.dot(C.T,F),C)
  return F

def twoe_MO_tran(Eri,C_1,C_2):
  #Transform one- and two-electron integrals to MO basis. The transformation of the 4-index array can be worked out by writing the 
  #basis transformation of a normal 2-D matrix as sums over the matrix elements. Input array assumed to be mulliken ordering (pq|rs).
  #Output integrals are in Dirac ordering <pr|qs>
  Eri_temp  = np.einsum('us,pqru->pqrs',C_2,Eri)
  Eri       = np.einsum('ur,pqus->pqrs',C_2,Eri_temp)
  Eri_temp  = np.einsum('uq,purs->pqrs',C_1,Eri)
  Eri       = np.einsum('up,uqrs->pqrs',C_1,Eri_temp)
  Eri = np.swapaxes(Eri,1,2) #Convert back to Dirac/Mulliken ordering 
  return Eri

def twoe_MO_tran_UHF_to_GHF(Eri,C_1,C_2,nocca,noccb,nbas):
  #Transform one- and two-electron integrals to MO basis. The transformation of the 4-index array can be worked out by writing the 
  #basis transformation of a normal 2-D matrix as sums over the matrix elements. Input array assumed to be mulliken ordering (pq|rs).
  #Output integrals are in Dirac ordering <pr|qs>
  import time
  nvirta = nbas-nocca
  nvirtb = nbas-noccb
  in1 = nocca
  in2 = in1 + noccb
  in3 = in2 + nvirta
#  Eri_temp  = np.einsum('us,pqru->pqrs',C_2,Eri)
#  Eri       = np.einsum('ur,pqus->pqrs',C_1,Eri_temp)
#  Eri_temp  = np.einsum('uq,purs->pqrs',C_2,Eri)
#  Eri       = np.einsum('up,uqrs->pqrs',C_1,Eri_temp)

  #We can accomplish the transformation more quickly if we use only non-zero blocks of the eigenvectors
  temp_1   = np.einsum('us,pqru->pqrs',C_2[:nbas,:in1],Eri[:,:,:,:nbas])
  temp_2   = np.append(temp_1,np.einsum('us,pqru->pqrs',C_2[nbas:,in1:in2],Eri[:,:,:,nbas:]),axis=3)
  temp_3   = np.append(temp_2,np.einsum('us,pqru->pqrs',C_2[:nbas,in2:in3],Eri[:,:,:,:nbas]),axis=3)
  Eri_temp = np.append(temp_3,np.einsum('us,pqru->pqrs',C_2[nbas:,in3:],Eri[:,:,:,nbas:]),axis=3)

  temp_1 =                  np.einsum('ur,pqus->pqrs',C_2[:nbas,:in1],   Eri_temp[:,:,:nbas,:])
  temp_2 = np.append(temp_1,np.einsum('ur,pqus->pqrs',C_2[nbas:,in1:in2],Eri_temp[:,:,nbas:,:]),axis=2)
  temp_3 = np.append(temp_2,np.einsum('ur,pqus->pqrs',C_2[:nbas,in2:in3],Eri_temp[:,:,:nbas,:]),axis=2)
  Eri    = np.append(temp_3,np.einsum('ur,pqus->pqrs',C_2[nbas:,in3:],   Eri_temp[:,:,nbas:,:]),axis=2)

  temp_1   =                  np.einsum('uq,purs->pqrs',C_1[:nbas,:in1],Eri[:,:nbas,:,:])
  temp_2   = np.append(temp_1,np.einsum('uq,purs->pqrs',C_1[nbas:,in1:in2],Eri[:,nbas:,:,:]),axis=1)
  temp_3   = np.append(temp_2,np.einsum('uq,purs->pqrs',C_1[:nbas,in2:in3],Eri[:,:nbas,:,:]),axis=1)
  Eri_temp = np.append(temp_3,np.einsum('uq,purs->pqrs',C_1[nbas:,in3:],   Eri[:,nbas:,:,:]),axis=1)

  temp_1 =                  np.einsum('up,uqrs->pqrs',C_1[:nbas,:in1],   Eri_temp[:nbas,:,:,:])
  temp_2 = np.append(temp_1,np.einsum('up,uqrs->pqrs',C_1[nbas:,in1:in2],Eri_temp[nbas:,:,:,:]),axis=0)
  temp_3 = np.append(temp_2,np.einsum('up,uqrs->pqrs',C_1[:nbas,in2:in3],Eri_temp[:nbas,:,:,:]),axis=0)
  Eri    = np.append(temp_3,np.einsum('up,uqrs->pqrs',C_1[nbas:,in3:],   Eri_temp[nbas:,:,:,:]),axis=0)

  Eri = np.swapaxes(Eri,1,2) #Convert to Dirac ordering 
  return Eri
