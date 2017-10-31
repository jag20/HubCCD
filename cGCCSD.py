import numpy as np
from scf import moUHF_to_GHF
#This routine implements complex generalized CCSD for the Hubbard Hamiltonian in N^5 scaling
#by exploiting the sparsity of the two-electron integrals for Hubbard. Derivation and factorization
#of the spin-orbital CCSD equations were worked out by Tom Henderson.
def ccsd(ham,ampfile="none"):
	if (ham.hamtype != 'Hubbard'):
		print("This routine should only be used for Hubbard in the spin-orbital basis.")
	if (ham.wfn_type == 'uhf'):
		print("converting UHF wavefunction to spin-orbital basis")
		ham.F, ham.Eri, ham.C = moUHF_to_GHF(ham.C_a,ham.C_b,ham.F_a,ham.F_b,ham.Eri_aa,ham.nocca,ham.noccb,ham.nbas)
		ham.nso = 2*ham.nbas
		ham.nocc  = ham.nocca + ham.noccb
		ham.nvirt = ham.nvirta + ham.nvirtb
		ham.wfn_type = 'ghf'

	if (ham.Eri.dtype ==float):
		print("Converting real integrals to complex")
		ham.Eri = ham.Eri.astype(np.complex)
		ham.F = ham.F.astype(np.complex)


	#Initialize/get amplitudes
	T2 = np.zeros([ham.nocc,ham.nocc,ham.nvirt,ham.nvirt],dtype=np.complex)
	T1 = np.zeros([ham.nocc,ham.nvirt],dtype=np.complex)

	#Build some initial intermediates. ham.C is AOxMO, with alpha in the first nbas/2 rows, followed by beta.
	C_up   =  ham.C[:ham.nbas,:]
	C_down =  ham.C[ham.nbas:,:]

	Cuij = np.zeros((ham.nbas,ham.nso,ham.nso),dtype=np.complex)
	Cabu = np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	C_up_up_pqu = np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	C_up_down_pqu= np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	C_down_up_pqu= np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	C_down_down_pqu= np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	for u in range(ham.nbas):
		for p in range(ham.nso):
			for q in range(ham.nso):
				Cuij[u,p,q] = C_up[u,p]*C_down[u,q] - C_down[u,p]*C_up[u,q] 
				Cabu[p,q,u] = (np.conj(C_up[u,p])*np.conj(C_down[u,q])
							- np.conj(C_down[u,p])*np.conj(C_up[u,q]))
				C_up_up_pqu[p,q,u]     = np.conj(C_up[u,p])*C_up[u,q]
				C_up_down_pqu[p,q,u]   = np.conj(C_up[u,p])*C_down[u,q]
				C_down_up_pqu[p,q,u]   = np.conj(C_down[u,p])*C_up[u,q]
				C_down_down_pqu[p,q,u] = np.conj(C_down[u,p])*C_down[u,q]

	print("Beginning Complex GCCSD Iterations")

	#Effective Amplitude Intermediates
	Tau = T2 + np.einsum('ia,jb->ijab',T1,T1)  - np.einsum('ib,ja->ijab',T1,T1)
	Tau_iju = np.einsum('ijab,uab->iju',Tau,Cuij[:,ham.nocc:,ham.nocc:])
	Tau_uab = np.einsum('ijab,iju->uab',Tau,Cabu[:ham.nocc,:ham.nocc,:])
	T_up_up_jbu = np.einsum('ijab,iau->jbu',T2,C_up_up_pqu[:ham.nocc,ham.nocc:,:])
	T_up_down_jbu = np.einsum('ijab,iau->jbu',T2,C_up_down_pqu[:ham.nocc,ham.nocc:,:])
	T_down_up_jbu = np.einsum('ijab,iau->jbu',T2,C_down_up_pqu[:ham.nocc,ham.nocc:,:])
	T_down_down_jbu = np.einsum('ijab,iau->jbu',T2,C_down_down_pqu[:ham.nocc,ham.nocc:,:])
	Tau_up_iu = np.zeros((ham.nocc,ham.nbas),dtype=np.complex)
	Tau_down_iu = np.zeros((ham.nocc,ham.nbas),dtype=np.complex)
	T_up_iu = np.zeros((ham.nocc,ham.nbas),dtype=np.complex)
	T_down_iu = np.zeros((ham.nocc,ham.nbas),dtype=np.complex)
	for i in range(ham.nocc):
		for u in range(ham.nbas):
			Tau_up_iu[i,u]   = np.einsum('j,j',Tau_iju[i,:,u],np.conj(C_up[u,:ham.nocc]))
			Tau_down_iu[i,u] = np.einsum('j,j',Tau_iju[i,:,u],np.conj(C_down[u,:ham.nocc]))
			T_up_iu[i,u]   = np.einsum('a,a',T1[i,:],C_up[u,ham.nocc:])
			T_down_iu[i,u] = np.einsum('a,a',T1[i,:],C_down[u,ham.nocc:])

	Tau_up_ua   = np.zeros((ham.nbas,ham.nvirt),dtype=np.complex)
	Tau_down_ua = np.zeros((ham.nbas,ham.nvirt),dtype=np.complex)
	T_up_ua   = np.zeros((ham.nbas,ham.nvirt),dtype=np.complex)
	T_down_ua = np.zeros((ham.nbas,ham.nvirt),dtype=np.complex)
	for u in range(ham.nbas):
		for a in range(ham.nvirt):
			Tau_up_ua[u,a]   = np.einsum('b,b',Tau_uab[u,a,:],C_up[u,ham.nocc:])
			Tau_down_ua[u,a] = np.einsum('b,b',Tau_uab[u,a,:],C_down[u,ham.nocc:])
			T_up_ua[u,a]   = np.einsum('i,i',T1[:,a],np.conj(C_up[u,:ham.nocc]))
			T_down_ua[u,a] = np.einsum('i,i',T1[:,a],np.conj(C_down[u,:ham.nocc]))
	T1_up_up_u     = np.zeros((ham.nbas),dtype=np.complex)
	T1_up_down_u   = np.zeros((ham.nbas),dtype=np.complex)
	T1_down_up_u   = np.zeros((ham.nbas),dtype=np.complex)
	T1_down_down_u = np.zeros((ham.nbas),dtype=np.complex)
	for u in range(ham.nbas):
		T1_up_up_u[u]   = np.einsum('ia,ia',C_up_up_pqu[:ham.nocc,ham.nocc:,u],T1)
		T1_up_down_u[u] = np.einsum('ia,ia',C_up_down_pqu[:ham.nocc,ham.nocc:,u],T1)
		T1_down_up_u[u] = np.einsum('ia,ia',C_down_up_pqu[:ham.nocc,ham.nocc:,u],T1)
		T1_down_down_u[u]   = np.einsum('ia,ia',C_down_down_pqu[:ham.nocc,ham.nocc:,u],T1)

	#Intermediates for the singles equations
	J_kc  = np.einsum('kcu,u->kc',C_up_up_pqu[:ham.nocc,ham.nocc:,:],T1_down_down_u)
	J_kc -= np.einsum('kcu,u->kc',C_up_down_pqu[:ham.nocc,ham.nocc:,:],T1_down_up_u)
	J_kc -= np.einsum('kcu,u->kc',C_down_up_pqu[:ham.nocc,ham.nocc:,:],T1_up_down_u)
	J_kc += np.einsum('kcu,u->kc',C_down_down_pqu[:ham.nocc,ham.nocc:,:],T1_down_down_u)
	J_kc *= ham.U

	J_ac  = np.einsum('uc,ua->ac',C_up[:,ham.nocc:],Tau_down_ua)
	J_ac -= np.einsum('uc,ua->ac',C_down[:,ham.nocc:],Tau_up_ua)
	J_ac *= ham.U

	J_ki  = np.einsum('uk,iu->ki',np.conj(C_up[:,:ham.nocc]),Tau_down_iu)
	J_ki -= np.einsum('uk,iu->ki',np.conj(C_down[:,:ham.nocc]),Tau_up_iu)
	J_ki *= ham.U




	#Get G1
	G1 = np.copy(ham.F[:ham.nocc:,ham.nocc:])
	F_offdiag = ham.F - np.diag(np.diag(ham.F))
	G1 += np.einsum('ac,ic->ia',F_offdiag[ham.nocc:,ham.nocc:],T1)
	G1 -= np.einsum('ki,ka->ia',F_offdiag[:ham.nocc,:ham.nocc],T1)
	G1 -= np.einsum('kc,ikac->ia',ham.F[:ham.nocc,ham.nocc:],(np.einsum('ic,ka->ikac',T1,T1) - T2))
	G1 += np.einsum('kc,ikac->ia',J_kc,(np.einsum('ic,ka->ikac',T1,T1) - T2))
	G1 -= 0.5e0*np.einsum('ic,ac->ia',T1,J_ac)
	G1 -= 0.5e0*np.einsum('ka,ki->ia',T1,J_ki)
	G1 += 0.5e0*ham.U*(np.einsum('ua,iu->ia',np.conj(C_up[:,ham.nocc:]),Tau_down_iu) 
       -  np.einsum('ua,iu->ia',np.conj(C_down[:,ham.nocc:]),Tau_up_iu))
	G1 -= 0.5e0*ham.U*(np.einsum('ui,ua->ia',C_up[:,:ham.nocc],Tau_down_ua) 
       - np.einsum('ui,ua->ia',C_down[:,:ham.nocc],Tau_up_ua))
	G1 += ham.U*(np.einsum('aiu,u->ia',C_up_up_pqu[ham.nocc:,:ham.nocc,:],T1_down_down_u)
	   -  np.einsum('aiu,u->ia',C_up_down_pqu[ham.nocc:,:ham.nocc,:],T1_down_up_u)
	   -  np.einsum('aiu,u->ia',C_down_up_pqu[ham.nocc:,:ham.nocc,:],T1_up_down_u)
	   +  np.einsum('aiu,u->ia',C_down_down_pqu[ham.nocc:,:ham.nocc,:],T1_down_down_u))


	#Get G2

 
 
 
 
 
 
 
 
 
	print("Done")
 
