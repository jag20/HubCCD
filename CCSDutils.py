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
	Tik = np.einsum('ck,ic->ik',F[nocc:,:nocc],T1)
	G -= np.einsum('ik,kjab->ijab',Tik,T2)
	G += np.einsum('jk,kiab->ijab',Tik,T2)
	Tca = np.einsum('ck,ka->ca',F[nocc:,:nocc],T1)
	G -= np.einsum('ca,ijcb->ijab',Tca,T2)
	G += np.einsum('cb,ijca->ijab',Tca,T2)
	Tidak = np.einsum('cdak,ic->idak',Eri[nocc:,nocc:,nocc:,:nocc],T1)
	G    += np.einsum('idak,kjdb->ijab',Tidak,T2)
	G    -= np.einsum('jdak,kidb->ijab',Tidak,T2)
	G    -= np.einsum('idbk,kjda->ijab',Tidak,T2)
	G    += np.einsum('jdbk,kida->ijab',Tidak,T2)
	Tical = np.einsum('ickl,ka->ical',Eri[:nocc,nocc:,:nocc,:nocc],T1)
	G    -= np.einsum('ical,ljcb->ijab',Tical,T2)
	G    += np.einsum('jcal,licb->ijab',Tical,T2)
	G    += np.einsum('icbl,ljca->ijab',Tical,T2)
	G    -= np.einsum('jcbl,lica->ijab',Tical,T2)
	Tcdab = np.einsum('cdkb,ka->cdab',Eri[nocc:,nocc:,:nocc,nocc:],T1)
	G    -= 0.5e0*np.einsum('cdab,ijcd->ijab',Tcdab,T2)
	Tcdba = np.einsum('cdka,kb->cdba',Eri[nocc:,nocc:,:nocc,nocc:],T1)
	G    += 0.5e0*np.einsum('cdba,ijcd->ijab',Tcdba,T2)
	#p. 308
	G += 0.5e0*np.einsum('cjkl,ic,klab->ijab',Eri[nocc:,:nocc,:nocc,:nocc],T1,T2) #Turns out we can do multiple contractions at once
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
	G += np.einsum('cdkb,ic,ka,jd->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T1,T1)
	G -= np.einsum('cdka,ic,kb,jd->ijab',Eri[nocc:,nocc:,:nocc,nocc:],T1,T1,T1)
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

