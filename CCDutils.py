import numpy as np
#This module contains the functions necessary for doing CCD in the RHF and spin-orbital basis
#as well as some CCD-based theories.


##RHF-based utilities
def RHFCCD(F,Eri,T,nocc,nbas,niter,variant="ccd"):
 #This function gets the right hand side of the rhf-based CCD equations. (The G in HT=G)
 #Spin-summation of the amplitude equations (only one set of t_ij^ab amplitudes) is from Scuseria, et al., JCP 86(5), 2881 (1987)
 #Terms quadratic in T are divided up into Ladder, Ring and Mosaic diagrams. See Bulik, Henderson and Scuseria, JCTC 11(7), 3171 (2015)
 #Diagrams have coefficients alpha and beta from parametrized CCD. See Nooijen, JCP 133, 184109 (2010)
 #We also implement spin-attenuated CCD (attCCD). See. Gomez, Henderson and Scuseria, Mol. Phys. onlin 23 Mar 2017.
 # CCD variants
 # variant = "ccd"   (the default, standard CCD)
 # 	   = "att"   (spin-attenuated CCD)
 # 	   = "lin"   (linearized CCD, i.e. no quadratic terms)
 #         = "acpq"  (ACPQ)
 #         = "acpd45"  (ACP-D45 from Paldus, etc.)
  variant = variant.lower()

  if (variant == "acpq") or (variant == "acpd45"):
    alpha = 1.0
    beta = 0.0
  else:
    alpha = 1.0
    beta  = 1.0

  nvirt = nbas-nocc
  c2 = (1.0e0/2.0e0)*(3.0e0/5.0e0)
  

  #Linear terms
  G = Lin(Eri,T,nocc)
  if (variant == "lin"):
     #Linearized Coupled Cluster
    return G

  else:
  #Get Quadratic Terms  
  #Ladder
    L = Ladder(Eri, T, nocc)
  
    #Rings
    R = Rings(Eri, T, nocc)
  
    #Mosaics and Mosaic p-h conjugates
    M = Mosaics(Eri, T, nocc)
    Mph = Mosaicsph(Eri, T, nocc)

    G += 0.5e0*(1.0e0 + alpha)*M + alpha*L + beta*(Mph + R)

    if (variant == "att"):
    #Do suhf attenuation of spin collective mode (attCCD) Gomez, Henderson and Scuseria, Mol. Phys. onlin 23 Mar 2017.
      G += Attenuate(Eri,T, nocc, nvirt,attnum=1,c2=3.0/10.0)


#    if (variant == "acpq"):
#    # don't forget 9* the triplet ladder for ACPQ
#      G += 4*L - 4*np.swapaxes(L,2,3)
#
    return G
    

def RCCDEn(Eri,T,nocc):
  #Spin-summed RHF-basis CCD energy
  vbar = (2*Eri - np.swapaxes(Eri,0,1))[nocc:,nocc:,:nocc,:nocc]
  eccd = np.einsum('abij,ijab',vbar,T)
  return eccd


def Lin(Eri,T,nocc):
  G = np.copy(Eri[:nocc,:nocc,nocc:,nocc:])
  Tbar = 2*T-np.swapaxes(T,2,3)
  G += np.einsum('klab,ijkl->ijab',T,Eri[:nocc,:nocc,:nocc,:nocc])
  G += np.einsum('ijcd,cdab->ijab',T,Eri[nocc:,nocc:,nocc:,nocc:])
  G += np.einsum('ikac,cjkb->ijab',Tbar,Eri[nocc:,:nocc,:nocc,nocc:])
  G += np.einsum('kjcb,icak->ijab',Tbar,Eri[:nocc,nocc:,nocc:,:nocc])
  G -= np.einsum('ikac,cjbk->ijab',T,Eri[nocc:,:nocc,nocc:,:nocc])
  G -= np.einsum('ikcb,cjak->ijab',T,Eri[nocc:,:nocc,nocc:,:nocc])
  G -= np.einsum('kjcb,icka->ijab',T,Eri[:nocc,nocc:,:nocc,nocc:])
  G -= np.einsum('kjac,ickb->ijab',T,Eri[:nocc,nocc:,:nocc,nocc:])
  return G

def Ladder(Eri, T, nocc):
  Jcdab = np.einsum('klab,cdkl->cdab',T, Eri[nocc:,nocc:,:nocc,:nocc])
  L = np.einsum('ijcd,cdab->ijab',T, Jcdab)
  return L
  

def Rings(Eri, T, nocc):
  vbar = (2*Eri - np.swapaxes(Eri,0,1))[nocc:,nocc:,:nocc,:nocc]
  Jdblj = np.einsum('kjcb,cdkl->dblj',T, vbar)
  R = np.einsum('ilad,dblj->ijab',(2*T - np.swapaxes(T,2,3)),Jdblj)
  Jdbjl = np.einsum('jkcb,cdkl->dbjl',T, vbar)
  R -= np.einsum('ilad,dbjl->ijab',T,Jdbjl)
  Jdbjl = np.einsum('jkcb,cdkl->dbjl',T, Eri[nocc:,nocc:,:nocc,:nocc])
  R += np.einsum('ilda,dbjl->ijab',T,Jdbjl)
  #This last term is actually from the cross-rings
  Jadlj = np.einsum('kjac,cdlk->adlj',T, Eri[nocc:,nocc:,:nocc,:nocc])
  R += np.einsum('ildb,adlj->ijab',T,Jadlj)
  return R

def Mosaics(Eri, T, nocc):
  vbar = (2*Eri - np.swapaxes(Eri,0,1))[nocc:,nocc:,:nocc,:nocc]
  Jki = np.einsum('cdkl,ilcd->ki', -vbar, T)
  M = np.einsum('ki,kjab->ijab',Jki, T)
  M += np.einsum('kj,ikab->ijab',Jki, T)
  return M

def Mosaicsph(Eri, T, nocc):
  vbar = (2*Eri - np.swapaxes(Eri,0,1))[nocc:,nocc:,:nocc,:nocc]
  #vbar = -1*(2*Eri - np.swapaxes(Eri,0,1))[nocc:,nocc:,:nocc,:nocc]
  #Jca = np.einsum('cdkl,klad->ca', vbar, T)
  Jca = np.einsum('cdkl,klad->ca', -vbar, T)
  Mph = np.einsum('ca,ijcb->ijab',Jca, T)
  Mph += np.einsum('cb,ijac->ijab',Jca, T)
  return Mph

def Attenuate(Eri,T, nocc, nvirt,attnum=1,c2=3.0/10.0):
  #Attenuated the spin collective mode 
  mdim = nocc*nvirt
  G = np.zeros(np.shape(T))
  K = np.swapaxes(T - 2.0e0*np.swapaxes(T,2,3),1,2)
  U = K.reshape((nocc*nvirt,nocc*nvirt))
  Uc  = np.zeros((nocc*nvirt,nocc*nvirt))
  Unc = np.zeros((nocc*nvirt,nocc*nvirt))
  #get spin collective modes
  l, V = np.linalg.eigh(U)
  idx = l.argsort()[::-1]
  l = l[idx]
  V = V[:,idx]
  attfact = 2.0e0*c2-1.0e0
  for i in range(attnum):
    lmat = np.zeros((mdim,mdim))
    lmat[i,i] = l[i]
    Uc = np.dot(V,np.dot(lmat,V.T))
    Uc = np.swapaxes(Uc.reshape((nocc,nvirt,nocc,nvirt)),1,2)
    K = -1.0e0/3.0e0*(Uc + 2.0e0*np.swapaxes(Uc,0,1))

    #Ladder
    L = Ladder(Eri, K, nocc)
    #Rings
    R = Rings(Eri, K, nocc)
    #Mosaics and Mosaic p-h conjugates
    M = Mosaics(Eri, K, nocc)
    Mph = Mosaicsph(Eri, K, nocc)

    #"unlinked" piece
    vbar = (2*Eri - np.swapaxes(Eri,0,1))[nocc:,nocc:,:nocc,:nocc]
    eccd = np.einsum('abij,ijab',vbar,K)
    G  += attfact*(M + L + Mph + R + K*eccd)
  return G

      
##GHF-based utilities
def GHFCCD(F,Eri,T,nocc,nbas,niter,variant="ccd"):
 #This function gets the right hand side of the ghf-based CCD equations. (The G in HT=G)
 #Terms quadratic in T are divided up into Ladder, Ring and Mosaic diagrams. See Bulik, Henderson and Scuseria, JCTC 11(7), 3171 (2015). Note the type on the sign of the mosaic terms in the reference.
 #Diagrams have coefficients alpha and beta from parametrized CCD. See Nooijen, JCP 133, 184109 (2010)
 # CCD variants
 # variant = "ccd"   (the default, standard CCD)
 # 	       = "patt"   (pair-attenuated CCD) Gomez, Henderson and Scuseria, Mol. Phys. onlin 23 Mar 2017.
 #   	   = "lin"   (linearized CCD, i.e. no quadratic terms)
 #         = "acpq"  (ACPQ)
  variant = variant.lower()
#  print("variant = ", variant)
  nvirt = nbas-nocc

  if (variant == "acpq"):
    alpha = 1.0
    beta = 0.0
  else:
    alpha = 1.0
    beta  = 1.0

  #Linear terms
  G = soLin(F,Eri,T,nocc)
  if (variant == "lin"):
     #Linearized Coupled Cluster
    return G

  else:
  #Get Quadratic Terms  
  #Ladder
    L = soLadder(Eri, T, nocc)
#  
#    #Rings
    R = soRings(Eri, T, nocc)
#  
#    #Mosaics and Mosaic p-h conjugates
    M   = soMosaics(Eri, T, nocc)
    Mph = soMosaicsph(Eri, T, nocc)
#
    G += 0.5e0*(1.0e0 + alpha)*M + alpha*L + beta*(Mph + R)

  if (variant == "patt"):
#    print("attenuating")
    G += pAttenuate(Eri,T, nocc, nvirt)

  return G
    
def soLin(F,Eri,T,nocc):
  G = np.copy(Eri[:nocc,:nocc,nocc:,nocc:])
  G += 1.0/2.0*np.einsum('klab,ijkl->ijab',T,Eri[:nocc,:nocc,:nocc,:nocc])
  G += 1.0/2.0*np.einsum('ijcd,cdab->ijab',T,Eri[nocc:,nocc:,nocc:,nocc:])
  G += np.einsum('ikac,cjkb->ijab',T,Eri[nocc:,:nocc,:nocc,nocc:])
  G += np.einsum('kjcb,cika->ijab',T,Eri[nocc:,:nocc,:nocc,nocc:])
  G -= np.einsum('ikbc,cjka->ijab',T,Eri[nocc:,:nocc,:nocc,nocc:])
  G -= np.einsum('kjca,cikb->ijab',T,Eri[nocc:,:nocc,:nocc,nocc:])
  #Get off-diagonal Fock terms if we're in a non-canonical basis
  tol = 1.0e-07
  F_offdiag = F - np.diag(np.diag(F))
  if np.amax(abs(F_offdiag) > tol):
    G += np.einsum('bc,ijac->ijab',F_offdiag[nocc:,nocc:],T)
    G += np.einsum('ac,ijcb->ijab',F_offdiag[nocc:,nocc:],T)
    G -= np.einsum('kj,ikab->ijab',F_offdiag[:nocc,:nocc],T)
    G -= np.einsum('ki,kjab->ijab',F_offdiag[:nocc,:nocc],T)

  return G

def soLadder(Eri, T, nocc):
    cdab = 1.0/4.0*np.einsum('klab,cdkl->cdab',T,Eri[nocc:,nocc:,:nocc,:nocc])
    return np.einsum('ijcd,cdab->ijab',T,cdab)

def soRings(Eri, T, nocc):
    jkbc = np.einsum('jlbd,cdkl->jkbc',T,Eri[nocc:,nocc:,:nocc,:nocc])
    R = np.einsum('ikac,jkbc->ijab',T,jkbc) 
#    tau = np.einsum('jlad,cdkl->jkac',T,Eri[nocc:,nocc:,:nocc,:nocc])
    R -= np.einsum('ikbc,jkac->ijab',T,jkbc) 
    return R

def soMosaics(Eri, T, nocc):
    M   = -1.0/2.0*np.einsum('cdkl,ilcd,kjab->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T,T)
    M  += 1.0/2.0*np.einsum('cdkl,ljcd,ikab->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T,T)
    return M

def soMosaicsph(Eri, T, nocc):
    Mph  = -1.0/2.0*np.einsum('cdkl,klad,ijcb->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T,T)
    Mph += 1.0/2.0*np.einsum('cdkl,kldb,ijac->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T,T)
    return Mph

def pAttenuate(Eri,T, nocc, nvirt,attnum=1,c2=1.0/4.0):
  #Attenuated the pairing collective mode 
  mdim = nocc*nocc
  ndim = nvirt*nvirt
  Uc  = np.zeros((mdim,ndim))
  #get pairing collective mode
  Umat = T.reshape(mdim,ndim)
  M, s, V = np.linalg.svd(Umat, full_matrices = True)
  attfact = 2.0e0*c2-1.0e0
  for i in range(attnum):
    smat = np.zeros((mdim,ndim))
    smat[i,i] = s[i]
    Uc = np.dot(M,np.dot(smat,V))
    K = np.reshape(Uc,(nocc,nocc,nvirt,nvirt))

    #Ladder
    L = soLadder(Eri, K, nocc)
    #Rings
    R = soRings(Eri, K, nocc)
    #Mosaics and Mosaic p-h conjugates
    M = soMosaics(Eri, K, nocc)
    Mph = soMosaicsph(Eri, K, nocc)

    #"unlinked" piece
    eccd = 1.0/4.0*np.einsum('ijab,abij',K,Eri[nocc:,nocc:,:nocc,:nocc])
    G  = attfact*(M + L + Mph + R + K*eccd)
  return G


def GCCDEn(Eri,T,nocc):
  #Spin-orbital T2 contribution to the CC energy
  eccd = 1.0/4.0*np.einsum('ijab,abij',T,Eri[nocc:,nocc:,:nocc,:nocc])
  return eccd


##DIIS extrapolation and equation-solving routines for both RHF and GHF CCD.
def solveccd(F,G,T,nocc,nvirt,x=4.0):
  Tnew = np.zeros(np.shape(T))
  for i in range(nocc):
    for j in range(nocc):
      for a in range(nvirt):
        aa = a + nocc
        for b in range(nvirt):
          bb = b + nocc
          d = (F[i,i] + F[j,j] - F[aa,aa] - F[bb,bb])
          Tnew[i,j,a,b] = G[i,j,a,b]/d
  #Damp amplitudes to improve convergence
  return(Tnew/x + T*(x-1.0)/x)

def diis_setup(nocc,nvirt):
  #use direct inversion of the iterative subspace (Pulay Chem Phys Lett 73(390), 1980) to extrapolate CC amplitudes.
  #This function sets up the various arrays we need for the extrapolation.
  diis_start = 15
  diis_dim = 4
  Errors  = np.zeros([diis_dim,nocc,nocc,nvirt,nvirt])
  Ts      = np.zeros([diis_dim,nocc,nocc,nvirt,nvirt])
  Err_vec = np.zeros([nocc,nocc,nvirt,nvirt])
  return diis_start, diis_dim, Errors, Ts, Err_vec

def diis(diis_start,diis_dim,iteration,Errors,Ts,Told,Err_vec):
  #use direct inversion of the iterative subspace (Pulay Chem Phys Lett 73(390), 1980) to accelerate convergence.
  #This function performs the actual extrapolation

  if (iteration > (diis_start + diis_dim)):
    #extrapolate the amplitudes if we're sufficiently far into the CC iterations

    #update error and amplitudes for next DIIS cycle. We DON'T want to store the extrapolated amplitudes in self.Ts
    Errors = np.roll(Errors,-1,axis=0)
    Errors[-1,:,:,:,:] = Err_vec
    Ts = np.roll(Ts,-1,axis=0)
    Ts [-1,:,:,:,:] = Told

    #solve the DIIS  Bc = l linear equation
    B = np.zeros((diis_dim+1,diis_dim+1))
    B[:,-1] = -1
    B[-1,:] = -1
    B[-1,-1] = 0
    B[:-1,:-1] = np.einsum('iklab,jklab->ij',Errors,Errors)
    l =  np.zeros(diis_dim+1)
    l[-1] = -1
    c = np.linalg.solve(B,l)

    T = np.einsum('q,qijab->ijab',c[:-1], Ts)
    

  elif (iteration > (diis_start)):
    #Fill the diis arrays until we have gone enough cycles to extrapolate
    count = iteration - diis_start - 1
    Errors[count,:,:,:,:] = Err_vec
    Ts[count:,:,:,:] = Told
    T = np.copy(Told)

  else:
    T = np.copy(Told)

  return T, Errors, Ts

def get_Err(F,G,T,nocc,nvirt):
  #Calculate the residual for the CC equations at a given value of T amplitudes
  Err_vec = np.zeros((nocc,nocc,nvirt,nvirt))
  for i in range(nocc):
    for j in range(nocc):
      for a in range(nvirt):
        aa = a + nocc
        for b in range(nvirt):
          bb = b + nocc
          Err_vec[i,j,a,b] = G[i,j,a,b]-(F[i,i] + F[j,j] - F[aa,aa] - F[bb,bb])*T[i,j,a,b]
  error = np.amax(np.absolute(Err_vec))
  return error, Err_vec


