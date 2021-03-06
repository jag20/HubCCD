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

  #Setup DIIS

  diis_start, diis_dim, Errors, F_as, F_bs, Err_vec = diis_setup(ham.nbas)

  P_a = np.copy(ham.P_a)
  P_b = np.copy(ham.P_b)
#  P_a = np.zeros(ham.P_a.shape)
#  P_b = np.zeros(ham.P_b.shape)


  #Read density matrix from file if we have it
  if ((denfile.lower() != "none") and os.path.isfile(denfile)):
    print("Reading previous density matrix")
    with open(denfile, 'rb') as f:
      P_a = pickle.load(f)
      P_b = pickle.load(f)

#  F_a, F_b = buildFs_uhf(P_a,P_b,ham.OneH,ham.Eri)
  #Set some values and do the UHF iteration
  error = 1.0
  tol = 1.0e-8
  eold = 1.0
  niter = 1
  x = 4.0

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

	#DIIS
    normerr, Err_vec = get_Err(Err_vec,F_a,F_b,P_a,P_b,ham.nbas)
    print("err = ", normerr)
#    F_a, F_b, Errors, F_as, F_bs =  diis(diis_start,diis_dim,niter,Errors,F_as,F_bs,F_a,F_b,Err_vec)


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
  ham.escf = escf 

  #Build final charge density matrix and get transformation to NO basis
  P = 0.5e0*(P_a+P_b)
  occ_nums, AO_to_NO = np.linalg.eigh(P)
  idx = (-occ_nums).argsort()
  occ_nums = occ_nums[idx]
  AO_to_NO = AO_to_NO[:,idx]

  if (denfile.lower() != "none"):
      with open(denfile, "wb") as f:
        pickle.dump(P_a, f)
        pickle.dump(P_b, f)

  ham.Pa = np.copy(P_a)
  ham.Pb = np.copy(P_b)
  return F_a, F_b, C_a, C_b, AO_to_NO






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
#  Eri_temp  = np.einsum('us,pqru->pqrs',C_2,Eri)
#  Eri       = np.einsum('ur,pqus->pqrs',C_2,Eri_temp)
#  Eri_temp  = np.einsum('uq,purs->pqrs',C_1,Eri)
#  Eri       = np.einsum('up,uqrs->pqrs',C_1,Eri_temp)
#  Eri = np.swapaxes(Eri,1,2) #Convert back to Dirac/Mulliken ordering 
  Eri_temp  = np.einsum('us,pqru->pqrs',C_2,Eri)
  Eri       = np.einsum('ur,pqus->pqrs',np.conj(C_2),Eri_temp)
  Eri_temp  = np.einsum('uq,purs->pqrs',C_1,Eri)
  Eri       = np.einsum('up,uqrs->pqrs',np.conj(C_1),Eri_temp)
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

def diis_setup(nbas):
  #use direct inversion of the iterative subspace (Pulay Chem Phys Lett 73(390), 1980) to extrapolate Fock matrix for SCF.
  #This function sets up the various arrays we need for the extrapolation.
  diis_start = 10
  diis_dim = 4
  Errors  = np.zeros([diis_dim,2*nbas*nbas])
  F_as    = np.zeros([diis_dim,nbas,nbas])
  F_bs    = np.zeros([diis_dim,nbas,nbas])
  Err_vec = np.zeros(2*nbas*nbas)
  return diis_start, diis_dim, Errors, F_as, F_bs,Err_vec



def diis(diis_start,diis_dim,iteration,Errors,F_as,F_bs,F_aold,F_bold,Err_vec):
  #use direct inversion of the iterative subspace (Pulay Chem Phys Lett 73(390), 1980) to accelerate convergence.
  #This function performs the actual extrapolation. Same structure as the version for the T-amplitudes (ccdutils), but for SCF.

  if (iteration > (diis_start + diis_dim)):
    #extrapolate the amplitudes if we're sufficiently far into the SCF iterations

    #update error and amplitudes for next DIIS cycle. 
    Errors = np.roll(Errors,-1,axis=0)
    Errors[-1,:] = Err_vec
    F_as = np.roll(F_as,-1,axis=0)
    F_bs = np.roll(F_bs,-1,axis=0)
    F_as [-1,:,:] = F_aold
    F_bs [-1,:,:] = F_bold

    #solve the DIIS  Bc = l linear equation
    B = np.zeros((diis_dim+1,diis_dim+1))
    B[:,-1] = -1
    B[-1,:] = -1
    B[-1,-1] = 0
    B[:-1,:-1] = np.einsum('ik,jk->ij',Errors,Errors)
    l =  np.zeros(diis_dim+1)
    l[-1] = -1
    c = np.linalg.solve(B,l)

    F_a = np.einsum('q,qij->ij',c[:-1], F_as)
    F_b = np.einsum('q,qij->ij',c[:-1], F_bs)
    

  elif (iteration > (diis_start)):
    #Fill the diis arrays until we have gone enough cycles to extrapolate
    count = iteration - diis_start - 1
    Errors[count,:] = Err_vec
    F_as[count,:,:] = F_aold
    F_bs[count,:,:] = F_bold
    F_a = np.copy(F_aold)
    F_b = np.copy(F_bold)

  else:
    F_a = np.copy(F_aold)
    F_b = np.copy(F_bold)

  return F_a, F_b, Errors, F_as, F_bs 

def get_Err(Err_vec,F_A,F_B,P_A,P_B,nbas):
  #At convergence, [F,P] = 0, so our error vector will be this
  #commutator.
  Err_vec = np.zeros((2*nbas*nbas))
  Err_vec[:nbas*nbas] = np.reshape(np.dot(F_A,P_A) - np.dot(P_A,F_A),nbas*nbas) 
  Err_vec[nbas*nbas:] = np.reshape(np.dot(F_B,P_B) - np.dot(P_B,F_B),nbas*nbas)
  error = np.linalg.norm((Err_vec))
  return error, Err_vec


