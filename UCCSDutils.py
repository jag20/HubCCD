import numpy as np
#Here we define some extra utilities needed for spin-summed, unrestricted CCSD

def solveccd(F_a,F_b,G,T,nocca,noccb,nvirta,nvirtb,x=4.0):
  #Solve for opposite-spin amplitudes
  Tnew = np.zeros(np.shape(T))
  for i in range(nocca):
    for j in range(noccb):
      for a in range(nvirta):
        aa = a + nocca
        for b in range(nvirtb):
          bb = b + noccb
          d = (F_a[i,i] + F_b[j,j] - F_a[aa,aa] - F_b[bb,bb])
          Tnew[i,j,a,b] = G[i,j,a,b]/d
  #Damp amplitudes to improve convergence
  return(Tnew/x + T*(x-1.0)/x)

def Ecorr(F_a,F_b,Eri_aa,Eri_ab,Eri_bb,T2_aa,T2_ab,T2_bb,T1_a,T1_b,nocca,noccb):
    #unrestricted, spin-summation of the CCSD correlation energy
    #CCD piece
	Eaa  = 0.25e0*np.einsum('ijab,abij',T2_aa,Eri_aa[nocca:,nocca:,:nocca,:nocca])
	Ebb = 0.25e0*np.einsum('ijab,abij',T2_bb,Eri_bb[noccb:,noccb:,:noccb,:noccb])
	Eab = np.einsum('ijab,abij',T2_ab,Eri_ab[nocca:,noccb:,:nocca,:noccb])
#	print("E2aa =", Eaa)
#	print("E2ab =", Eab)
#	print("E2bb =", Ebb)
	E2 = Eaa + Ebb + Eab
	#linear in singles
	E1 = np.einsum('ia,ai',T1_a,F_a[nocca:,:nocca])
	E1 += np.einsum('ia,ai',T1_b,F_b[noccb:,:noccb])
    
	#quadratic in singles
	E1 += 0.5e0*np.einsum('ia,jb,abij',T1_a,T1_a,Eri_aa[nocca:,nocca:,:nocca,:nocca])
	E1 += 0.5e0*np.einsum('ia,jb,abij',T1_a,T1_b,Eri_ab[nocca:,noccb:,:nocca,:noccb])
	E1 += 0.5e0*np.einsum('ia,jb,baji',T1_b,T1_a,Eri_ab[nocca:,noccb:,:nocca,:noccb])
	E1 += 0.5e0*np.einsum('ia,jb,abij',T1_b,T1_b,Eri_bb[noccb:,noccb:,:noccb,:noccb])
#	print("E1 =", E1)
	return E1+E2

def diis_setup(diis_start,diis_dim,nocca,noccb,nvirta,nvirtb):
	#use direct inversion of the iterative subspace (Pulay Chem Phys Lett 73(390), 1980) to extrapolate CC amplitudes.
	#This function sets up the various arrays we need for the extrapolation.
	Errors  = np.zeros([diis_dim,nocca,noccb,nvirta,nvirtb])
	Ts      = np.zeros([diis_dim,nocca,noccb,nvirta,nvirtb])
	Err_vec = np.zeros([nocca,noccb,nvirta,nvirtb])
	return Errors, Ts, Err_vec

def get_Err(F_a,F_b,G,T,nocca,noccb,nvirta,nvirtb):
  #Calculate the residual for the CC equations at a given value of T amplitudes
  Err_vec = np.zeros((nocca,noccb,nvirta,nvirtb))
  for i in range(nocca):
    for j in range(noccb):
      for a in range(nvirta):
        aa = a + nocca
        for b in range(nvirtb):
          bb = b + noccb
          Err_vec[i,j,a,b] = G[i,j,a,b]-(F_a[i,i] + F_b[j,j] - F_a[aa,aa] - F_b[bb,bb])*T[i,j,a,b]
  error = np.amax(np.absolute(Err_vec))
  return error, Err_vec


def get_non_canon(F_a_offdiag,F_b_offdiag,T2_aa,T2_ab,T2_bb,T1_a,T1_b,nocca,noccb):
	#Get contractions over off-diagonal Fock terms if we're in a non-canonical basis (Fock not diagonal)
	G2_aa  = np.einsum('bc,ijac->ijab',F_a_offdiag[nocca:,nocca:],T2_aa)
	G2_aa += np.einsum('ac,ijcb->ijab',F_a_offdiag[nocca:,nocca:],T2_aa)
	G2_aa -= np.einsum('kj,ikab->ijab',F_a_offdiag[:nocca,:nocca],T2_aa)
	G2_aa -= np.einsum('ki,kjab->ijab',F_a_offdiag[:nocca,:nocca],T2_aa)
	
	G2_ab  = np.einsum('bc,ijac->ijab',F_b_offdiag[noccb:,noccb:],T2_ab)
	G2_ab += np.einsum('ac,ijcb->ijab',F_a_offdiag[nocca:,nocca:],T2_ab)
	G2_ab -= np.einsum('kj,ikab->ijab',F_b_offdiag[:noccb,:noccb],T2_ab)
	G2_ab -= np.einsum('ki,kjab->ijab',F_a_offdiag[:nocca,:nocca],T2_ab)
	
	G2_bb  = np.einsum('bc,ijac->ijab',F_b_offdiag[noccb:,noccb:],T2_bb)
	G2_bb += np.einsum('ac,ijcb->ijab',F_b_offdiag[noccb:,noccb:],T2_bb)
	G2_bb -= np.einsum('kj,ikab->ijab',F_b_offdiag[:noccb,:noccb],T2_bb)
	G2_bb -= np.einsum('ki,kjab->ijab',F_b_offdiag[:noccb,:noccb],T2_bb)
	
	G1_a  = np.einsum('ca,ic->ia',F_a_offdiag[nocca:,nocca:],T1_a)
	G1_a -= np.einsum('ik,ka->ia',F_a_offdiag[:nocca,:nocca],T1_a)
	
	G1_b  = np.einsum('ca,ic->ia',F_b_offdiag[noccb:,noccb:],T1_b)
	G1_b -= np.einsum('ik,ka->ia',F_b_offdiag[:noccb,:noccb],T1_b)
	
	return [G1_a, G1_b, G2_aa, G2_ab, G2_bb]


def get_res2(T,F_a,F_b,G,nocca,noccb,nvirta,nvirtb):
  #Calculate the residual for the CC equations at a given value of T amplitudes
  T = np.reshape(T,(nocca,noccb,nvirta,nvirtb))
  G = np.reshape(G,(nocca,noccb,nvirta,nvirtb))
  F_a = np.reshape(F_a,(nocca+nvirta,nocca+nvirta))
  F_b = np.reshape(F_b,(noccb+nvirtb,noccb+nvirtb))
  Err_vec = np.zeros((nocca,noccb,nvirta,nvirtb))
  for i in range(nocca):
    for j in range(noccb):
      for a in range(nvirta):
        aa = a + nocca
        for b in range(nvirtb):
          bb = b + noccb
          Err_vec[i,j,a,b] = G[i,j,a,b]-(F_a[i,i] + F_b[j,j] - F_a[aa,aa] - F_b[bb,bb])*T[i,j,a,b]
  return np.einsum('ijab,ijab',Err_vec,Err_vec)

def get_res1(T,F,G,nocc,nvirt):
  #Calculate the residual for the CC equations at a given value of T amplitudes
  T = np.reshape(T,(nocc,nvirt))
  G = np.reshape(G,(nocc,nvirt))
  F = np.reshape(F,(nocc+nvirt,nocc+nvirt))
  Err_vec = np.zeros((nocc,nvirt))
  for i in range(nocc):
      for a in range(nvirt):
        aa = a + nocc
        Err_vec[i,a] = G[i,a]-(F[i,i] - F[aa,aa])*T[i,a]
  return np.einsum('ia,ia',Err_vec,Err_vec)


def SolveT1_CG(F,T,G,nocc,nvirt):
 	#I'm having trouble gettin Python's Conjugate gradient routines to work with solving HT=G, since I
    #Don't want to actually build the full H, so I'll code up a simple version here,
	#almost verbatim from Trefethen and Bau, p. 294.

	#return if we have zero solution
	checktol=1.0e-10
	if np.einsum('ia,ia',G,G) < checktol:
		return np.zeros(T.shape)
	#initialize vectors
	x = np.copy(T)
	r0 = np.copy(G)
	p = np.copy(r0)

	niter = 0
	tol = 1.0e-12
	error = tol*50
	while (error > tol):
		niter += 1
		#Get A*x
		Ax = -np.einsum('ca,ic->ia',F[nocc:,nocc:],p)
		Ax += np.einsum('ki,ka->ia',F[:nocc,:nocc],p)
		#step length
		alpha =  np.einsum('ij,ij',r0,r0)/np.einsum('ij,ij',p,Ax)
		#New solution
		x += alpha*p
		#residual
		r1 = r0 - alpha*Ax
		#Improvement factor beta
		beta = np.einsum('ij,ij',r1,r1)/(np.einsum('ij,ij',r0,r0))
		#new search direction
		p = r1 + beta*p
		#reset some values
		r0 = np.copy(r1)
		error = np.einsum('ia,ia',r0,r0)
		print("niter = ", niter, "error = ", error)
	return x

		
		
def T1LHS(F,T,nocc,nvirt):
	#contract T1 with Fock
	LHS = np.zeros(np.shape(T1),dtype=T1.dtype)
	for i in range(nocc):
		for a in range(nvirt):
			aa = a + nocc
			d = (F[i,i] - F[aa,aa])
			LHS[i,a] = T1[i,a]*d
		






