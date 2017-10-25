import numpy as np
import CCDutils
#This module contains the functions necessary for doing CCSD in the spin-orbital basis


##Spin-orbital-based utilities
def CCSDdoubles(F,Eri,T2,T1,nocc,nbas,variant):
	#Get the right hand side of the spinorbital CCSD singles equations. p. 307-308 of Bartlett and Shavitt
	niter = 1
	#p.307
    #Get CCD contribution
	G = CCDutils.GHFCCD(F,Eri,T2,nocc,nbas,niter,variant)
#	return G
	G += np.einsum('cjab,ic->ijab',Eri[nocc:,:nocc,nocc:,nocc:],T1)
	G -= np.einsum('ciab,jc->ijab',Eri[nocc:,:nocc,nocc:,nocc:],T1)
	G -= np.einsum('ijkb,ka->ijab',Eri[:nocc,:nocc,:nocc,nocc:],T1)
	G += np.einsum('ijka,kb->ijab',Eri[:nocc,:nocc,:nocc,nocc:],T1)
	G -= np.einsum('ck,ic,kjab->ijab',F[nocc:,:nocc],T1,T2)
	G += np.einsum('ck,jc,kiab->ijab',F[nocc:,:nocc],T1,T2)
	G -= np.einsum('ck,ka,ijcb->ijab',F[nocc:,:nocc],T1,T2)
	G += np.einsum('ck,kb,ijca->ijab',F[nocc:,:nocc],T1,T2)
	G    += np.einsum('cdak,ic,kjdb->ijab',Eri[nocc:,nocc:,nocc:,:nocc],T1,T2)
	G    -= np.einsum('cdbk,ic,kjda->ijab',Eri[nocc:,nocc:,nocc:,:nocc],T1,T2)
	G    -= np.einsum('cdak,jc,kidb->ijab',Eri[nocc:,nocc:,nocc:,:nocc],T1,T2)
	G    += np.einsum('cdbk,jc,kida->ijab',Eri[nocc:,nocc:,nocc:,:nocc],T1,T2)
	G    -= np.einsum('ickl,ka,ljcb->ijab',Eri[:nocc,nocc:,:nocc,:nocc],T1,T2)
	G    += np.einsum('jckl,ka,licb->ijab',Eri[:nocc,nocc:,:nocc,:nocc],T1,T2)
	G    += np.einsum('ickl,kb,ljca->ijab',Eri[:nocc,nocc:,:nocc,:nocc],T1,T2)
	G    -= np.einsum('jckl,kb,lica->ijab',Eri[:nocc,nocc:,:nocc,:nocc],T1,T2)
	G    -= 0.5e0*np.einsum('cdkb,ka,ijcd->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T2)
	G    += 0.5e0*np.einsum('cdka,kb,ijcd->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T2)
	#p. 308
	G += 0.5e0*np.einsum('cjkl,ic,klab->ijab',Eri[nocc:,:nocc,:nocc,:nocc],T1,T2) 
	G -= 0.5e0*np.einsum('cikl,jc,klab->ijab',Eri[nocc:,:nocc,:nocc,:nocc],T1,T2) 
	G += np.einsum('cdka,kc,ijdb->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T2) 
	G -= np.einsum('cdkb,kc,ijda->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T2) 
	G -= np.einsum('cikl,kc,ljab->ijab',Eri[nocc:,:nocc,:nocc,:nocc],T1,T2) 
	G += np.einsum('cjkl,kc,liab->ijab',Eri[nocc:,:nocc,:nocc,:nocc],T1,T2) 
	G += np.einsum('cdab,ic,jd->ijab',Eri[nocc:,nocc:,nocc:,nocc:],T1,T1)
	G += np.einsum('ijkl,ka,lb->ijab',Eri[:nocc,:nocc,:nocc,:nocc],T1,T1)
	G -= np.einsum('cjkb,ic,ka->ijab',Eri[nocc:,:nocc,:nocc,nocc:],T1,T1)
	G += np.einsum('cikb,jc,ka->ijab',Eri[nocc:,:nocc,:nocc,nocc:],T1,T1)
	G += np.einsum('cjka,ic,kb->ijab',Eri[nocc:,:nocc,:nocc,nocc:],T1,T1)
	G -= np.einsum('cika,jc,kb->ijab',Eri[nocc:,:nocc,:nocc,nocc:],T1,T1)
	G += 0.5e0*np.einsum('cdkl,ic,jd,klab->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G += 0.5e0*np.einsum('cdkl,ka,lb,ijcd->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G -= np.einsum('cdkl,ic,ka,ljdb->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G += np.einsum('cdkl,jc,ka,lidb->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G += np.einsum('cdkl,ic,kb,ljda->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G -= np.einsum('cdkl,jc,kb,lida->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G -= np.einsum('cdkl,kc,id,ljab->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G += np.einsum('cdkl,kc,jd,liab->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G -= np.einsum('cdkl,kc,la,ijdb->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	G += np.einsum('cdkl,kc,lb,ijda->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T2)
	# -------------------------------
	#	G += np.einsum('cdkb,ic,ka,jd->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T1,T1) 
	#	G -= np.einsum('cdka,ic,kb,jd->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T1,T1) wrong sign in diagram D8a, p. 306
	#	of Shavitt and Bartlett. Compare to 
	G -= np.einsum('cdkb,ic,ka,jd->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T1,T1)
	G += np.einsum('cdka,ic,kb,jd->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T1,T1)
	#from Crawford and Schaefer, An Introduction to Coupled Cluster Theory, Wiley ...
	#-----------------------------------------------------------------------------
	G += np.einsum('cjkl,ic,ka,lb->ijab',Eri[nocc:,:nocc,:nocc,:nocc],T1,T1,T1)
	G -=np.einsum('cikl,jc,ka,lb->ijab',Eri[nocc:,:nocc,:nocc,:nocc],T1,T1,T1)
	G += np.einsum('cdkl,ic,jd,ka,lb->ijab',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T1,T1)

	return G

def CCSDsingles(F,Eri,T2,T1,nocc,nbas):
	#Get the right hand side of the spinorbital CCSD singles equations. p. 304 of Bartlett and Shavitt
	#Driver
	G = np.copy(F[:nocc,nocc:])
	#Terms involving only doubles
	G += np.einsum('kc,ikac->ia',F[:nocc,nocc:],T2)
	G += 0.5e0*np.einsum('cdak,ikcd->ia',Eri[nocc:,nocc:,nocc:,:nocc],T2)
	G -= 0.5e0*np.einsum('ickl,klac->ia',Eri[:nocc,nocc:,:nocc,:nocc],T2)
	#Linear term involving only singles
	G += np.einsum('icak,kc->ia',Eri[:nocc,nocc:,nocc:,:nocc],T1)
	#Mixed Terms 
	G -= 0.5e0*np.einsum('cdkl,ic,klad->ia',Eri[nocc:,nocc:,:nocc,:nocc],T1,T2)
	G -= 0.5e0*np.einsum('cdkl,ka,ilcd->ia',Eri[nocc:,nocc:,:nocc,:nocc],T1,T2)
	G += np.einsum('cdkl,kc,lida->ia',Eri[nocc:,nocc:,:nocc,:nocc],T1,T2)
	#higher-order terms involving only singles
	G -= np.einsum('ck,ic,ka->ia',F[nocc:,:nocc],T1,T1)
	G += np.einsum('cdak,ic,kd->ia',Eri[nocc:,nocc:,nocc:,:nocc],T1,T1) 
	G -= np.einsum('ickl,ka,lc->ia',Eri[:nocc,nocc:,:nocc,:nocc],T1,T1)
	G -= np.einsum('cdkl,ic,ka,ld->ia',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1,T1)
	#Don't forget other non-canonical terms
	tol = 1.0e-07
	F_offdiag = F - np.diag(np.diag(F))
	if np.amax(abs(F_offdiag) > tol):
		G += np.einsum('ca,ic->ia',F_offdiag[nocc:,nocc:],T1)
		G -= np.einsum('ik,ka->ia',F_offdiag[:nocc,:nocc],T1)
	return G

def CCSDsingles_fact(F,Eri,T2,T1,nocc,nbas):
    #build intermediates according to Stanton et al. JCP 94(6) 1991
    F_diag = np.diag(np.diag(F))
    Tau_tilde = T2 + 0.50e0*(np.einsum('ia,jb->ijab',T1,T1)-np.einsum('ib,ja->ijab',T1,T1))
    Fae = F[nocc:,nocc:] - F_diag[nocc:,nocc:] 
    Fae -= 0.5e0*(np.einsum('em,ma->ea',F[nocc:,:nocc],T1))
    Fae += np.einsum('mf,fema->ea',T1,Eri[nocc:,nocc:,:nocc,nocc:])
    Fae -= 0.5e0*np.einsum('mnaf,efmn->ea',Tau_tilde,Eri[nocc:,nocc:,:nocc,:nocc])

    Fmi = F[:nocc,:nocc] - F_diag[:nocc,:nocc] 
    Fmi += 0.5e0*(np.einsum('em,ie->im',F[nocc:,:nocc],T1))
    Fmi += np.einsum('ne,iemn->im',T1,Eri[:nocc,nocc:,:nocc,:nocc])
    Fmi += 0.5e0*np.einsum('inef,efmn->im',Tau_tilde,Eri[nocc:,nocc:,:nocc,:nocc])

    Fme = F[nocc:,:nocc] + np.einsum('nf,efmn->em',T1,Eri[nocc:,nocc:,:nocc,:nocc])

    #contract T with intermediates to get RHS of singles equation. (eq 1 in Stanton reference)
    G = F[:nocc,nocc:] + np.einsum('ie,ea->ia',T1,Fae)
    G -= np.einsum('ma,im->ia',T1,Fmi)
    G += np.einsum('imae,em->ia',T2,Fme)
    G -= np.einsum('nf,ifna->ia',T1,Eri[:nocc,nocc:,:nocc,nocc:])
    G -= 0.5e0*np.einsum('imef,efma->ia',T2,Eri[nocc:,nocc:,:nocc,nocc:])
    G -= 0.5e0*np.einsum('mnae,einm->ia',T2,Eri[nocc:,:nocc,:nocc,:nocc])
    return G

def CCSDdoubles_fact(F,Eri,T2,T1,nocc,nbas):
    #build intermediates according to Stanton et al. JCP 94(6) 1991
    F_diag = np.diag(np.diag(F))
    Tau_tilde = T2 + 0.50e0*(np.einsum('ia,jb->ijab',T1,T1)-np.einsum('ib,ja->ijab',T1,T1))
    Tau  = T2 + np.einsum('ia,jb->ijab',T1,T1) - np.einsum('ib,ja->ijab',T1,T1)
    #2-index intermediates
    Fae = F[nocc:,nocc:] - F_diag[nocc:,nocc:] 
    Fae -= 0.5e0*(np.einsum('em,ma->ea',F[nocc:,:nocc],T1))
    Fae += np.einsum('mf,fema->ea',T1,Eri[nocc:,nocc:,:nocc,nocc:])
    Fae -= 0.5e0*np.einsum('mnaf,efmn->ea',Tau_tilde,Eri[nocc:,nocc:,:nocc,:nocc])

    Fmi = F[:nocc,:nocc] - F_diag[:nocc,:nocc] 
    Fmi += 0.5e0*(np.einsum('em,ie->im',F[nocc:,:nocc],T1))
    Fmi += np.einsum('ne,iemn->im',T1,Eri[:nocc,nocc:,:nocc,:nocc])
    Fmi += 0.5e0*np.einsum('inef,efmn->im',Tau_tilde,Eri[nocc:,nocc:,:nocc,:nocc])

    Fme = F[nocc:,:nocc] + np.einsum('nf,efmn->em',T1,Eri[nocc:,nocc:,:nocc,:nocc])
    #4-index intermediates
    Wijmn = Eri[:nocc,:nocc,:nocc,:nocc] + np.einsum('je,iemn->ijmn',T1,Eri[:nocc,nocc:,:nocc,:nocc])
    Wijmn -= np.einsum('ie,jemn->ijmn',T1,Eri[:nocc,nocc:,:nocc,:nocc])
    Wijmn += 0.25e0*np.einsum('ijef,efmn->ijmn',Tau,Eri[nocc:,nocc:,:nocc,:nocc])

    Wefab = Eri[nocc:,nocc:,nocc:,nocc:] - np.einsum('mb,efam->efab',T1,Eri[nocc:,nocc:,nocc:,:nocc])
    Wefab += np.einsum('ma,efbm->efab',T1,Eri[nocc:,nocc:,nocc:,:nocc])
    Wefab += 0.25e0*np.einsum('mnab,efmn->efab',Tau,Eri[nocc:,nocc:,:nocc,:nocc])

    Wejmb = Eri[nocc:,:nocc,:nocc,nocc:] + np.einsum('jf,efmb->ejmb',T1,Eri[nocc:,nocc:,:nocc,nocc:])
    Wejmb -= np.einsum('nb,ejmn->ejmb',T1,Eri[nocc:,:nocc,:nocc,:nocc])
    tau1 = np.einsum('jf,nb->jnfb',T1,T1)
    Wejmb -= np.einsum('jnfb,efmn->ejmb',(0.5e0*T2+tau1),Eri[nocc:,nocc:,:nocc,:nocc])
    #contract T with intermediates to get RHS of singles equation. (eq 2 in Stanton reference)
    
    G = Eri[:nocc,:nocc,nocc:,nocc:] + np.einsum('ijae,eb->ijab',T2,Fae) - 0.5e0*np.einsum('ijae,mb,em->ijab',T2,T1,Fme)
    G -= (np.einsum('ijbe,ea->ijab',T2,Fae) - 0.5e0*np.einsum('ijbe,ma,em->ijab',T2,T1,Fme))
    G -= (np.einsum('imab,jm->ijab',T2,Fmi) + 0.5e0*np.einsum('imab,je,em->ijab',T2,T1,Fme))
    G += (np.einsum('jmab,im->ijab',T2,Fmi) + 0.5e0*np.einsum('jmab,ie,em->ijab',T2,T1,Fme))
    G += 0.5e0*(np.einsum('mnab,ijmn->ijab',Tau,Wijmn) + np.einsum('ijef,efab->ijab',Tau,Wefab))
    G += (np.einsum('imae,ejmb->ijab',T2,Wejmb) - np.einsum('ie,ma,ejmb->ijab',T1,T1,Eri[nocc:,:nocc,:nocc,nocc:]))
    G -= (np.einsum('jmae,eimb->ijab',T2,Wejmb) - np.einsum('je,ma,eimb->ijab',T1,T1,Eri[nocc:,:nocc,:nocc,nocc:]))
    G -= (np.einsum('imbe,ejma->ijab',T2,Wejmb) - np.einsum('ie,mb,ejma->ijab',T1,T1,Eri[nocc:,:nocc,:nocc,nocc:]))
    G += (np.einsum('jmbe,eima->ijab',T2,Wejmb) - np.einsum('je,mb,eima->ijab',T1,T1,Eri[nocc:,:nocc,:nocc,nocc:]))
    G += np.einsum('ie,ejab->ijab',T1,Eri[nocc:,:nocc,nocc:,nocc:])  
    G -= np.einsum('je,eiab->ijab',T1,Eri[nocc:,:nocc,nocc:,nocc:])  
    G -= np.einsum('ma,ijmb->ijab',T1,Eri[:nocc,:nocc,:nocc,nocc:])  
    G += np.einsum('mb,ijma->ijab',T1,Eri[:nocc,:nocc,:nocc,nocc:])  


    return G




def solveccs(F,G1,T1,nocc,nvirt,x=4.0):
	#solve singles amplitude equations
	T1new = np.zeros(np.shape(T1))
	for i in range(nocc):
		for a in range(nvirt):
			aa = a + nocc
			d = (F[i,i] - F[aa,aa])
			T1new[i,a] = G1[i,a]/d
	#Damp amplitudes to improve convergence
	return(T1new/x + T1*(x-1.0)/x)

def GCCSEn(F,Eri,T1,nocc):
	#Spin-orbital T1 contribution to the CC energy
	eccs = np.einsum('abij,ia,jb',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1)
	eccs -= np.einsum('abij,ib,ja',Eri[nocc:,nocc:,:nocc,:nocc],T1,T1)
	eccs *= 0.25e0
	eccs += np.einsum('ai,ia',F[nocc:,:nocc],T1)
	return eccs

#DIIS for singles
def diis_singles_setup(nocc,nvirt,diis_start,diis_dim):
  #use direct inversion of the iterative subspace (Pulay Chem Phys Lett 73(390), 1980) to extrapolate CC amplitudes.
  #This function sets up the various arrays we need for the extrapolation.
  Errors  = np.zeros([diis_dim,nocc,nvirt])
  Ts      = np.zeros([diis_dim,nocc,nvirt])
  Err_vec = np.zeros([nocc,nvirt])
  return Errors, Ts, Err_vec

def get_singles_Err(F,G,T,nocc,nvirt):
  #Calculate the residual for the CC equations at a given value of T amplitudes
  Err_vec = np.zeros((nocc,nvirt))
  for i in range(nocc):
      for a in range(nvirt):
        aa = a + nocc
        Err_vec[i,a] = G[i,a]-(F[i,i] - F[aa,aa] )*T[i,a]
  error = np.amax(np.absolute(Err_vec))
  return error, Err_vec

def diis_singles(diis_start,diis_dim,iteration,Errors,Ts,Told,Err_vec):
  #use direct inversion of the iterative subspace (Pulay Chem Phys Lett 73(390), 1980) to accelerate convergence.
  #This function performs the actual extrapolation

  if (iteration > (diis_start + diis_dim)):
    #extrapolate the amplitudes if we're sufficiently far into the CC iterations

    #update error and amplitudes for next DIIS cycle. We DON'T want to store the extrapolated amplitudes in self.Ts
    Errors = np.roll(Errors,-1,axis=0)
    Errors[-1,:,:] = Err_vec
    Ts = np.roll(Ts,-1,axis=0)
    Ts [-1,:,:] = Told

    #solve the DIIS  Bc = l linear equation
    B = np.zeros((diis_dim+1,diis_dim+1))
    B[:,-1] = -1
    B[-1,:] = -1
    B[-1,-1] = 0
    B[:-1,:-1] = np.einsum('ika,jka->ij',Errors,Errors)
    l =  np.zeros(diis_dim+1)
    l[-1] = -1
    c = np.linalg.solve(B,l)

    T = np.einsum('q,qia->ia',c[:-1], Ts)
    

  elif (iteration > (diis_start)):
    #Fill the diis arrays until we have gone enough cycles to extrapolate
    count = iteration - diis_start - 1
    Errors[count,:,:] = Err_vec
    Ts[count:,:] = Told
    T = np.copy(Told)

  else:
    T = np.copy(Told)

  return T, Errors, Ts
