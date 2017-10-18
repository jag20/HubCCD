import numpy as np
import CCDutils
#This module contains the functions necessary for doing CCSD in the spin-orbital basis


##Spin-orbital-based utilities
def CCSDdoubles(F,Eri,T2,T1,nocc,nbas):
	niter = 1
    #Get CCD contribution
	G = CCDutils.GHFCCD(F,Eri,T2,nocc,nbas,niter,variant="ccd")

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
	Tidkl = np.einsum('cdkl,ic->idkl',Eri[nocc:,nocc:,:nocc,:nocc],T1)
	G -= 0.5e0*np.einsum('idkl,klad->ia',Tidkl,T2)
	Tcdal = np.einsum('cdkl,ka->cdal',Eri[nocc:,nocc:,:nocc,:nocc],T1)
	G -= 0.5e0*np.einsum('cdal,ilcd->ia',Tcdal,T2)
	Tdl = np.einsum('cdkl,kc->dl',Eri[nocc:,nocc:,:nocc,:nocc],T1)
	G += np.einsum('dl,lida->ia',Tdl,T2)
	#higher-order terms involving only singles
	Tik = np.einsum('ck,ic->ik',F[nocc:,:nocc],T1)
	G -= np.einsum('ik,ka->ia',Tik,T1)
	Tidak = np.einsum('cdak,ic->idak',Eri[nocc:,nocc:,nocc:,:nocc],T1)
	G += np.einsum('idak,kd->ia',Tidak,T1)
	Tical = np.einsum('ickl,ka->ical',Eri[:nocc,nocc:,:nocc,:nocc],T1)
	G -= np.einsum('ical,lc->ia',Tical,T1)
	Tidal = np.einsum('idkl,ka->idal',Tidkl,T1)
	G -= np.einsum('idal,ld->ia',Tidal,T1)
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
	Gbj   = np.einsum('abij,ia->bj',Eri[nocc:,nocc:,:nocc,:nocc],T1)
	eccs  = np.einsum('bj,jb',Gbj,T1)
	Gaj   = np.einsum('abij,ib->aj',Eri[nocc:,nocc:,:nocc,:nocc],T1)
	eccs  -= np.einsum('aj,ja',Gaj,T1)
	eccs *= 0.25e0
	eccs += np.einsum('ai,ia',F[nocc:,:nocc],T1)
	return eccs

