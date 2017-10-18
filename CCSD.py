from CCSDutils import *
import CCDutils
import os
import pickle
def ccsd(ham,ampfile="none"):
	error = 1.0
	tol = 1.0e-07
	eold = 1.0
#read amplitudes from file if present to improve convergence
	if (os.path.isfile(ampfile)):
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
		G1 = CCSDsingles(ham.F,ham.Eri,T2,T1,ham.nocc,ham.nbas)
		G2 = CCSDdoubles(ham.F,ham.Eri,T2,T1,ham.nocc,ham.nbas)

		#solve H
		T1 = solveccs(ham.F,G1,T1,ham.nocc,ham.nvirt)
		T2 = CCDutils.solveccd(ham.F,G2,T2,ham.nocc,ham.nvirt)

		#get energies
		E2 = CCDutils.GCCDEn(ham.Eri,T2,ham.nocc)
		E1 = GCCSEn(ham.F,ham.Eri,T1,ham.nocc)
		ecorr = E1 + E2
		error = np.abs(ecorr-eold)
		print("Energy = ", ecorr, "error = ", error)
		eold = ecorr 
	
	
