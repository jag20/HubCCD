from CCSDutils import *
import CCDutils
import os
import pickle
from scf import UHF_to_GHF, MO_tran

def ccsd(ham,ampfile="none"):
	#We need to convert integrals to spin-orbital basis if our meanfield is RHF
	if (ham.wfn_type == 'rhf'):
		print("converting RHF wavefunction to spin-orbital basis")

		#First transform back to AO basis
		ham.F, ham.Eri = MO_tran(ham.F,ham.Eri,np.linalg.inv(ham.C))
		F_a, F_b, C_a, C_b = np.copy(ham.F), np.copy(ham.F), np.copy(ham.C), np.copy(ham.C)
		nocca = ham.nocc
		noccb = ham.nocc
		F_GHF, Eri_GHF, C_GHF = UHF_to_GHF(C_a,C_b,F_a,F_b,ham.Eri,nocca,noccb,ham.nbas)
		ham.F, ham.Eri, ham.C = np.copy(F_GHF), np.copy(Eri_GHF), np.copy(C_GHF)
		ham.nbas = 2*ham.nbas
		ham.nocc = 2*ham.nocc
		ham.nvirt = 2*ham.nvirt
		
	error = 1.0
	tol = 1.0e-08
	eold = 1.0
#read amplitudes from file if present to improve convergence
	if ((ampfile != 'none') and(os.path.isfile(ampfile))):
		with open(ampfile, 'rb') as f:
			T2 = pickle.load(f)
			T1 = pickle.load(f)

		E2 = CCDutils.GCCDEn(Eri,T,nocc)
		E1 = GCCSEn(F,Eri,T1,nocc)
		ecorr = E1 + E2
		eold = ecorr
	else:
		T2 = np.zeros([ham.nocc,ham.nocc,ham.nvirt,ham.nvirt])
		T1 = np.zeros([ham.nocc,ham.nvirt])
		eold = 0.0e0

	while (error > tol):
		#build RHS
#		G1 = CCSDsingles(ham.F,ham.Eri,T2,T1,ham.nocc,ham.nbas)
		G2 = CCSDdoubles(ham.F,ham.Eri,T2,T1,ham.nocc,ham.nbas)

		#solve H
#		T1 = solveccs(ham.F,G1,T1,ham.nocc,ham.nvirt)
		T2 = CCDutils.solveccd(ham.F,G2,T2,ham.nocc,ham.nvirt)

		#get energies
#		E1 = GCCSEn(ham.F,ham.Eri,T1,ham.nocc)
		E1 = 0.0e0
		E2 = CCDutils.GCCDEn(ham.Eri,T2,ham.nocc)
		ecorr = E1 + E2
		error = np.abs(ecorr-eold)
		print("Energy = ", ecorr, "error = ", error)
		eold = ecorr 


	ham.ecorr = ecorr
	
	
