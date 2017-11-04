#Module cGSSD implements complex generalized CCSD for the Hubbard Hamiltonian in N^5 scaling
#by exploiting the sparsity of the two-electron integrals for Hubbard. Derivation and factorization
#of the spin-orbital CCSD equations were worked out by Tom Henderson.

import numpy as np
#import cCCutils
import CCDutils
import CCSDutils
from scf import onee_MO_tran

def ccsd(ham,ampfile="none"):
	if (ham.hamtype != 'Hubbard'):
		print("This routine should only be used for Hubbard in the spin-orbital basis.")
	if (ham.wfn_type == 'uhf'):
		print("converting One-electron Integrals to spin-orbital basis")
		
		#We build the 2-e integrals on the fly, 
		#so will only build the spin-orbital Fock matrix and MO coefficients.
		Fa_ao   = onee_MO_tran(ham.F_a,np.linalg.inv(ham.C_a))
		Fb_ao   = onee_MO_tran(ham.F_b,np.linalg.inv(ham.C_b))
		nvirta = ham.nbas-ham.nocca
		nvirtb = ham.nbas-ham.noccb
		C = np.zeros([ham.nbas*2,ham.nbas*2])
		F = np.zeros([ham.nbas*2,ham.nbas*2])
		#Build spin-orbital Mo coefficients
		C[:ham.nbas,:ham.nocca] = ham.C_a[:,:ham.nocca]
		C[ham.nbas:,ham.nocca:(ham.nocca+ham.noccb)] = ham.C_b[:,:ham.noccb]
		C[:ham.nbas,(ham.nocca+ham.noccb):(ham.nocca+ham.noccb+nvirta)]  = ham.C_a[:,ham.nocca:]
		C[ham.nbas:,(ham.nocca+ham.noccb+nvirta):] = ham.C_b[:,ham.noccb:]
		#build spinorbital fock
		F[:ham.nbas,:ham.nbas] = Fa_ao
		F[ham.nbas:,ham.nbas:] = Fb_ao
		F = onee_MO_tran(F,C)
		print(np.diag(F))
		ham.F = np.copy(F)
		ham.C = np.copy(C)

		ham.nso = 2*ham.nbas
		ham.nocc  = ham.nocca + ham.noccb
		ham.nvirt = ham.nvirta + ham.nvirtb
		ham.wfn_type = 'ghf'

	if (ham.F.dtype ==float):
		print("Converting real integrals to complex")
		ham.F = ham.F.astype(np.complex)
		ham.C = ham.C.astype(np.complex)


	#Initialize/get amplitudes
	T2 = np.zeros([ham.nocc,ham.nocc,ham.nvirt,ham.nvirt],dtype=np.complex)
	T1 = np.zeros([ham.nocc,ham.nvirt],dtype=np.complex)


	#set up for DIIS
	diis_start, diis_dim, Errors, T2s, Err_vec = CCDutils.diis_setup(ham.nocc,ham.nvirt,ham.F.dtype)
	T1Errors, T1s, T1Err_vec = CCSDutils.diis_singles_setup(ham.nocc,ham.nvirt,diis_start,diis_dim,ham.F.dtype)
	#Convert arrays to complex as necessary
#	Errors, T2s, Err_vec = Errors.astype(np.complex), T2s.astype(np.complex), Err_vec.astype(np.complex)
#	T1Errors, T1s, T1Err_vec = T1Errors.astype(np.complex), T1s.astype(np.complex), T1Err_vec.astype(np.complex)

	#Build some initial intermediates. ham.C is AOxMO, with alpha in the first nbas/2 rows, followed by beta.
	C_up   =  ham.C[:ham.nbas,:]
	C_down =  ham.C[ham.nbas:,:]

	C_upq = np.zeros((ham.nbas,ham.nso,ham.nso),dtype=np.complex)
	C_pqu = np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	C_up_up_pqu = np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	C_up_down_pqu= np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	C_down_up_pqu= np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	C_down_down_pqu= np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)

	for u in range(ham.nbas):
		for p in range(ham.nso):
			for q in range(ham.nso):
				C_upq[u,p,q] = C_up[u,p]*C_down[u,q] - C_down[u,p]*C_up[u,q] 
				C_pqu[p,q,u] = (np.conj(C_up[u,p])*np.conj(C_down[u,q])
							- np.conj(C_down[u,p])*np.conj(C_up[u,q]))
				C_up_up_pqu[p,q,u]     = np.conj(C_up[u,p])*C_up[u,q]
				C_up_down_pqu[p,q,u]   = np.conj(C_down[u,p])*C_up[u,q]
				C_down_up_pqu[p,q,u]   = np.conj(C_up[u,p])*C_down[u,q]
				C_down_down_pqu[p,q,u] = np.conj(C_down[u,p])*C_down[u,q]

	print("Beginning Complex GCCSD Iterations")
	eold = 0.0e0
	error = 1.0
	tol = 1.0e-8
	damping = 2
	niter = 1
	while (error > tol):
		T2, Errors, T2s   = CCDutils.diis(diis_start,diis_dim,niter,Errors,T2s,T2,Err_vec)
		T1, T1Errors, T1s = CCSDutils.diis_singles(diis_start,diis_dim,niter,T1Errors,T1s,T1,T1Err_vec)
	
		#Effective Amplitude Intermediates
		Tau = T2 + np.einsum('ia,jb->ijab',T1,T1)  - np.einsum('ib,ja->ijab',T1,T1)
		Tau_uij = np.einsum('ijab,uab->uij',Tau,C_upq[:,ham.nocc:,ham.nocc:])
		Tau_abu = np.einsum('ijab,iju->abu',Tau,C_pqu[:ham.nocc,:ham.nocc,:])
		T_up_up_bju     = np.einsum('ijab,iau->bju',T2,C_up_up_pqu[:ham.nocc,ham.nocc:,:])
		T_up_down_bju   = np.einsum('ijab,iau->bju',T2,C_up_down_pqu[:ham.nocc,ham.nocc:,:])
		T_down_up_bju   = np.einsum('ijab,iau->bju',T2,C_down_up_pqu[:ham.nocc,ham.nocc:,:])
		T_down_down_bju = np.einsum('ijab,iau->bju',T2,C_down_down_pqu[:ham.nocc,ham.nocc:,:])
		T_up_up_aiu     = np.einsum('ijab,jbu->aiu',T2,C_up_up_pqu[:ham.nocc,ham.nocc:,:])
		T_up_down_aiu   = np.einsum('ijab,jbu->aiu',T2,C_up_down_pqu[:ham.nocc,ham.nocc:,:])
		T_down_up_aiu   = np.einsum('ijab,jbu->aiu',T2,C_down_up_pqu[:ham.nocc,ham.nocc:,:])
		T_down_down_aiu = np.einsum('ijab,jbu->aiu',T2,C_down_down_pqu[:ham.nocc,ham.nocc:,:])

		Tau_up_iu = np.zeros((ham.nocc,ham.nbas),dtype=np.complex)
		Tau_down_iu = np.zeros((ham.nocc,ham.nbas),dtype=np.complex)
		T_up_iu = np.zeros((ham.nocc,ham.nbas),dtype=np.complex)
		T_down_iu = np.zeros((ham.nocc,ham.nbas),dtype=np.complex)
		for i in range(ham.nocc):
			for u in range(ham.nbas):
				Tau_up_iu[i,u]   = np.einsum('j,j',Tau_uij[u,i,:],np.conj(C_up[u,:ham.nocc]))
				Tau_down_iu[i,u] = np.einsum('j,j',Tau_uij[u,i,:],np.conj(C_down[u,:ham.nocc]))
				T_up_iu[i,u]   = np.einsum('a,a',T1[i,:],C_up[u,ham.nocc:])
				T_down_iu[i,u] = np.einsum('a,a',T1[i,:],C_down[u,ham.nocc:])
	
		Tau_up_ua   = np.zeros((ham.nbas,ham.nvirt),dtype=np.complex)
		Tau_down_ua = np.zeros((ham.nbas,ham.nvirt),dtype=np.complex)
		T_up_ua   = np.zeros((ham.nbas,ham.nvirt),dtype=np.complex)
		T_down_ua = np.zeros((ham.nbas,ham.nvirt),dtype=np.complex)
		for u in range(ham.nbas):
			for a in range(ham.nvirt):
				Tau_up_ua[u,a]   = np.einsum('b,b',Tau_abu[a,:,u],C_up[u,ham.nocc:])
				Tau_down_ua[u,a] = np.einsum('b,b',Tau_abu[a,:,u],C_down[u,ham.nocc:])
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
		J_kc += np.einsum('kcu,u->kc',C_down_down_pqu[:ham.nocc,ham.nocc:,:],T1_up_up_u)
		J_kc *= ham.U
	
		J_ac  = np.einsum('uc,ua->ac',C_up[:,ham.nocc:],Tau_down_ua)
		J_ac -= np.einsum('uc,ua->ac',C_down[:,ham.nocc:],Tau_up_ua)
		J_ac *= ham.U
	
		J_ki  = np.einsum('uk,iu->ki',np.conj(C_up[:,:ham.nocc]),Tau_down_iu)
		J_ki -= np.einsum('uk,iu->ki',np.conj(C_down[:,:ham.nocc]),Tau_up_iu)
		J_ki *= ham.U
	
	
	
	
		#Get G1
		G1 = np.copy(ham.F[:ham.nocc,ham.nocc:])
		F_offdiag = ham.F - np.diag(np.diag(ham.F))
		G1 += np.einsum('ac,ic->ia',F_offdiag[ham.nocc:,ham.nocc:],T1)
		G1 -= np.einsum('ki,ka->ia',F_offdiag[:ham.nocc,:ham.nocc],T1)
		G1 -= np.einsum('kc,ikac->ia',ham.F[:ham.nocc,ham.nocc:],(np.einsum('ic,ka->ikac',T1,T1) - T2))
		G1 += np.einsum('kc,ikac->ia',J_kc,(np.einsum('ic,ka->ikac',T1,T1) + T2))
		G1 -= 0.5e0*np.einsum('ic,ac->ia',T1,J_ac)
		G1 -= 0.5e0*np.einsum('ka,ki->ia',T1,J_ki)
		G1 += 0.5e0*ham.U*(np.einsum('ua,iu->ia',np.conj(C_up[:,ham.nocc:]),Tau_down_iu) 
	       -  np.einsum('ua,iu->ia',np.conj(C_down[:,ham.nocc:]),Tau_up_iu))
		G1 -= 0.5e0*ham.U*(np.einsum('ui,ua->ia',C_up[:,:ham.nocc],Tau_down_ua) 
	       - np.einsum('ui,ua->ia',C_down[:,:ham.nocc],Tau_up_ua))
		G1 += ham.U*(np.einsum('aiu,u->ia',C_up_up_pqu[ham.nocc:,:ham.nocc,:],T1_down_down_u)
		   -  np.einsum('aiu,u->ia',C_up_down_pqu[ham.nocc:,:ham.nocc,:],T1_down_up_u)
		   -  np.einsum('aiu,u->ia',C_down_up_pqu[ham.nocc:,:ham.nocc,:],T1_up_down_u)
		   +  np.einsum('aiu,u->ia',C_down_down_pqu[ham.nocc:,:ham.nocc,:],T1_up_up_u))
	
	
		#Intermediates for doubles equations
		X_up_up_aiu     = C_up_up_pqu[ham.nocc:,:ham.nocc,:]     + 0.50e0*T_up_up_bju 
		X_up_down_aiu   = C_up_down_pqu[ham.nocc:,:ham.nocc,:]   + 0.50e0*T_up_down_bju 
		X_down_up_aiu   = C_down_up_pqu[ham.nocc:,:ham.nocc,:]   + 0.50e0*T_down_up_bju 
		X_down_down_aiu = C_down_down_pqu[ham.nocc:,:ham.nocc,:] + 0.50e0*T_down_down_bju 
#		X_up_up_aiu     = C_up_up_pqu[ham.nocc:,:ham.nocc,:]     + 0.50e0*T_up_up_aiu 
#		X_up_down_aiu   = C_up_down_pqu[ham.nocc:,:ham.nocc,:]   + 0.50e0*T_up_down_aiu 
#		X_down_up_aiu   = C_down_up_pqu[ham.nocc:,:ham.nocc,:]   + 0.50e0*T_down_up_aiu 
#		X_down_down_aiu = C_down_down_pqu[ham.nocc:,:ham.nocc,:] + 0.50e0*T_down_down_aiu 
		for a in range(ham.nvirt):
			aa = a + ham.nocc
			for i in range(ham.nocc):
				for u in range(ham.nbas):
					X_up_up_aiu[a,i,u] -= T_up_iu[i,u]*T_up_ua[u,a]
					X_up_up_aiu[a,i,u] += T_up_iu[i,u]*np.conj(C_up[u,aa])
					X_up_up_aiu[a,i,u] -= T_up_ua[u,a]*C_up[u,i]
	
					X_up_down_aiu[a,i,u] -= T_up_iu[i,u]*T_down_ua[u,a]
					X_up_down_aiu[a,i,u] += T_up_iu[i,u]*np.conj(C_down[u,aa])
					X_up_down_aiu[a,i,u] -= T_down_ua[u,a]*C_up[u,i]
	
					X_down_up_aiu[a,i,u] -= T_down_iu[i,u]*T_up_ua[u,a]
					X_down_up_aiu[a,i,u] += T_down_iu[i,u]*np.conj(C_up[u,aa])
					X_down_up_aiu[a,i,u] -= T_up_ua[u,a]*C_down[u,i]
	
					X_down_down_aiu[a,i,u] -= T_down_iu[i,u]*T_down_ua[u,a]
					X_down_down_aiu[a,i,u] += T_down_iu[i,u]*np.conj(C_down[u,aa])
					X_down_down_aiu[a,i,u] -= T_down_ua[u,a]*C_down[u,i]
	
	
		K_ad  = np.einsum('adu,u->ad',C_up_up_pqu[ham.nocc:,ham.nocc:,:],T1_down_down_u)
		K_ad -= np.einsum('adu,u->ad',C_up_down_pqu[ham.nocc:,ham.nocc:,:],T1_down_up_u)
		K_ad -= np.einsum('adu,u->ad',C_down_up_pqu[ham.nocc:,ham.nocc:,:],T1_up_down_u)
		K_ad += np.einsum('adu,u->ad',C_down_down_pqu[ham.nocc:,ham.nocc:,:],T1_up_up_u)
		K_ad -= 0.5e0*(np.einsum('ud,ua->ad',C_up[:,ham.nocc:],Tau_down_ua)
	         -  np.einsum('ud,ua->ad',C_down[:,ham.nocc:],Tau_up_ua))
		K_ad *= ham.U
		K_ad -= np.einsum('ld,la',ham.F[:ham.nocc,ham.nocc:],T1)
		
		K_li  = -np.einsum('liu,u->li', C_up_up_pqu[:ham.nocc,:ham.nocc,:],T1_down_down_u)
		K_li += np.einsum('liu,u->li',  C_up_down_pqu[:ham.nocc,:ham.nocc,:],T1_down_up_u)
		K_li += np.einsum('liu,u->li',  C_down_up_pqu[:ham.nocc,:ham.nocc,:],T1_up_down_u)
		K_li -= np.einsum('liu,u->li',C_down_down_pqu[:ham.nocc,:ham.nocc,:],T1_up_up_u)
		K_li -= 0.5e0*(np.einsum('ul,iu->li',np.conj(C_up[:,:ham.nocc]),Tau_down_iu)
	         -  np.einsum('ul,iu->li',np.conj(C_down[:,:ham.nocc]),Tau_up_iu))
		K_li *= ham.U
		K_li -= np.einsum('ld,id',ham.F[:ham.nocc,ham.nocc:],T1)
	
	
		Xuij = C_upq[:,:ham.nocc,:ham.nocc] + 0.50e0*Tau_uij 
		Xabu = C_pqu[ham.nocc:,ham.nocc:,:] + 0.50e0*Tau_abu 
		for u in range(ham.nbas):
			for i in range(ham.nocc):
				for j in range(ham.nocc):
					Xuij[u,i,j] += C_up[u,i]*T_down_iu[j,u]
					Xuij[u,i,j] -= C_down[u,i]*T_up_iu[j,u]
					Xuij[u,i,j] -= C_up[u,j]*T_down_iu[i,u]
					Xuij[u,i,j] += C_down[u,j]*T_up_iu[i,u]
			for a in range(ham.nvirt):
				aa = a + ham.nocc
				for b in range(ham.nvirt):
					bb = b + ham.nocc
					Xabu[a,b,u] -= np.conj(C_up[u,aa])*T_down_ua[u,b]
					Xabu[a,b,u] += np.conj(C_down[u,aa])*T_up_ua[u,b]
					Xabu[a,b,u] += np.conj(C_up[u,bb])*T_down_ua[u,a]
#					Xabu[a,b,u] += np.conj(C_down[u,bb])*T_up_ua[u,a]
#					Check with Tom about Notes, but I think this term should have the opposite sign
					Xabu[a,b,u] -= np.conj(C_down[u,bb])*T_up_ua[u,a]
	 
		#Get G2
		#Rings
		Rings  = np.einsum('aiu,bju->ijab',X_up_up_aiu,T_down_down_bju)
		Rings -= np.einsum('aiu,bju->ijab',X_up_down_aiu,T_down_up_bju)
		Rings -= np.einsum('aiu,bju->ijab',X_down_up_aiu,T_up_down_bju)
		Rings += np.einsum('aiu,bju->ijab',X_down_down_aiu,T_up_up_bju)
		Rings *= ham.U
	
		#contractions
		G2  = ham.U*np.einsum('uij,abu->ijab',Xuij,Xabu) 
		G2 += (Rings - np.swapaxes(Rings,2,3) - np.swapaxes(Rings,0,1) + np.swapaxes(np.swapaxes(Rings,0,1),2,3))
		G2 += np.einsum('ad,ijdb->ijab',K_ad,T2)
		G2 += np.einsum('bd,ijad->ijab',K_ad,T2)
		G2 += np.einsum('li,ljab->ijab',K_li,T2)
		G2 += np.einsum('lj,ilab->ijab',K_li,T2)
	 
	    #non-canonical terms
		G2 += np.einsum('ac,ijcb->ijab',F_offdiag[ham.nocc:,ham.nocc:],T2)
		G2 += np.einsum('bc,ijac->ijab',F_offdiag[ham.nocc:,ham.nocc:],T2)
		G2 -= np.einsum('ik,kjab->ijab',F_offdiag[:ham.nocc,:ham.nocc],T2)
		G2 -= np.einsum('jk,ikab->ijab',F_offdiag[:ham.nocc,:ham.nocc],T2)
	 
	 
		#Get error vecs (residuals HT-G)
		T2error, Err_vec   = CCDutils.get_Err(ham.F,G2,T2,ham.nocc,ham.nvirt)
		T1error, T1Err_vec = CCSDutils.get_singles_Err(ham.F,G1,T1,ham.nocc,ham.nvirt)
	 
	 
		#solve HT = G
		T2 = CCDutils.solveccd(ham.F,G2,T2,ham.nocc,ham.nvirt,x=damping)
		T1 = CCSDutils.solveccs(ham.F,G1,T1,ham.nocc,ham.nvirt,x=damping)
	
		#Get energy
		Tau = T2 + np.einsum('ia,jb->ijab',T1,T1)  - np.einsum('ib,ja->ijab',T1,T1)
		Tau_uij = np.einsum('ijab,uab->uij',Tau,C_upq[:,ham.nocc:,ham.nocc:])
		ecorr = 0.25*ham.U*np.einsum('iju,uij',C_pqu[:ham.nocc,:ham.nocc,:],Tau_uij)
		ecorr += np.einsum('ia,ia',ham.F[:ham.nocc,ham.nocc:],T1)
		error = np.abs(eold-ecorr)
		eold = ecorr
		print("Iteration ", niter, " Energy = ", ecorr, " Error = ", error)
		niter +=1

	ham.ecorr = ecorr
	 
