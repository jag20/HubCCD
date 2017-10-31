from CCSDutils import *
from UCCSDG2 import getccsdg2, getg2, ccsdenergy
from UCCSDG1 import getg1
import UCCSDutils
import pickle
import CCD
import CCDutils
import os

def ccsd(ham,ampfile="none",variant="ccd"):
	if (ham.wfn_type == 'rhf'):
		print("converting RHF wavefunction to UHF basis")
		ham.nocca = ham.nocc
		ham.noccb = ham.nocc
		ham.nvirta = ham.nvirt
		ham.nvirtb = ham.nvirt
		ham.F_a = np.copy(ham.F)
		ham.F_b = np.copy(ham.F)
		ham.Eri_aa = np.copy(ham.Eri)
		ham.Eri_ab = np.copy(ham.Eri)
		ham.Eri_bb = np.copy(ham.Eri)
		ham.Eri_aa = ham.Eri_aa - np.swapaxes(ham.Eri_aa,2,3)  #antisymmetrize
		ham.Eri_bb = ham.Eri_bb - np.swapaxes(ham.Eri_bb,2,3)  #antisymmetrize
	elif (ham.wfn_type == 'uhf'):
		ham.Eri_aa = ham.Eri_aa - np.swapaxes(ham.Eri_aa,2,3)  #antisymmetrize
		ham.Eri_bb = ham.Eri_bb - np.swapaxes(ham.Eri_bb,2,3)  #antisymmetrize
   
#read amplitudes from file if present to improve convergence
	if ((ampfile != 'none') and(os.path.isfile(ampfile))):
		with open(ampfile, 'rb') as f:
			T2_aa = pickle.load(f)
			T2_ab = pickle.load(f)
			T2_bb = pickle.load(f)
			T1_a = pickle.load(f)
			T1_b = pickle.load(f)
#
#		E2 = CCDutils.GCCDEn(ham.Eri,T2,ham.nocc)
#		E1 = GCCSEn(ham.F,ham.Eri,T1,ham.nocc)
#		ecorr = E1 + E2
#		eold = ecorr
#	else:
	#Need this for fortran routines
	va  = ham.nocca + 1
	vb  = ham.noccb + 1

	T2_aa = np.zeros([ham.nocca,ham.nocca,ham.nvirta,ham.nvirta],order='F')
	T2_ab = np.zeros([ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb],order='F')
	T2_bb = np.zeros([ham.noccb,ham.noccb,ham.nvirtb,ham.nvirtb],order='F')
	T1_a  = np.zeros([ham.nocca,ham.nvirta],order='F')
	T1_b  = np.zeros([ham.noccb,ham.nvirtb],order='F')
	G2_aa = np.zeros([ham.nocca,ham.nocca,ham.nvirta,ham.nvirta],order='F')
	G2_ab = np.zeros([ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb],order='F')
	G2_bb = np.zeros([ham.noccb,ham.noccb,ham.nvirtb,ham.nvirtb],order='F')
	G1_a  = np.zeros([ham.nocca,ham.nvirta],order='F')
	G1_b  = np.zeros([ham.noccb,ham.nvirtb],order='F')
	G1_a, G1_b  = getg1(T1_a,T1_b,T2_aa,T2_ab,T2_bb,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb, ham.nocca,ham.noccb,ham.nbas)
	G2_aa, G2_ab, G2_bb = getccsdg2(T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca,ham.noccb,ham.nbas)
	eold  = 0.0e0

	tol_off = 1.0e-08
	F_a_offdiag = ham.F_a - np.diag(np.diag(ham.F_a))
	if np.amax(abs(F_a_offdiag) > tol_off):
		print("Using a non-canonical basis")
		F_b_offdiag = ham.F_b - np.diag(np.diag(ham.F_b))
		offs = UCCSDutils.get_non_canon(F_a_offdiag,F_b_offdiag,T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.nocca,ham.noccb)
		G1_a  += offs[0]
		G1_b  += offs[1]
		G2_aa += offs[2]
		G2_ab += offs[3]
		G2_bb += offs[4]
	#Set up for CCSD iteration and DIIS. 
	diis_start, diis_dim, T2aaErrors, T2aas, T2aaErr_vec = CCDutils.diis_setup(ham.nocca,ham.nvirta)
	T2abErrors, T2abs, T2abErr_vec = UCCSDutils.diis_setup(diis_start,diis_dim,ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb)
	T2bbErrors, T2bbs, T2bbErr_vec = UCCSDutils.diis_setup(diis_start,diis_dim,ham.noccb,ham.noccb,ham.nvirtb,ham.nvirtb)
	T1aErrors, T1as, T1aErr_vec = diis_singles_setup(ham.nocca,ham.nvirta,diis_start,diis_dim)
	T1bErrors, T1bs, T1bErr_vec = diis_singles_setup(ham.noccb,ham.nvirtb,diis_start,diis_dim)

	niter = 1
	tol = 1.0e-8
	error = tol*50
	damping= 1


	print("Beginning UCCSD iteration")
	while (error > tol):
		T2_aa, T2aaErrors, T2aas   = CCDutils.diis(diis_start,diis_dim,niter,T2aaErrors,T2aas,T2_aa,T2aaErr_vec)
		T2_ab, T2abErrors, T2abs   = CCDutils.diis(diis_start,diis_dim,niter,T2abErrors,T2abs,T2_ab,T2abErr_vec)
		T2_bb, T2bbErrors, T2bbs   = CCDutils.diis(diis_start,diis_dim,niter,T2bbErrors,T2bbs,T2_bb,T2bbErr_vec)
		T1_a, T1aErrors, T1as = diis_singles(diis_start,diis_dim,niter,T1aErrors,T1as,T1_a,T1aErr_vec)
		T1_b, T1bErrors, T1bs = diis_singles(diis_start,diis_dim,niter,T1bErrors,T1bs,T1_b,T1bErr_vec)
   	#build RHS G
		G1_a, G1_b  = getg1(T1_a,T1_b,T2_aa,T2_ab,T2_bb,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb, ham.nocca,ham.noccb,ham.nbas)
		G2_aa, G2_ab, G2_bb = getccsdg2(T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca,ham.noccb,ham.nbas)
	#off-diagonal fock terms from LHS if non-canonical basis
		if np.amax(abs(F_a_offdiag) > tol_off):
			offs = UCCSDutils.get_non_canon(F_a_offdiag,F_b_offdiag,T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.nocca,ham.noccb)
			G1_a  += offs[0]
			G1_b  += offs[1]
			G2_aa += offs[2]
			G2_ab += offs[3]
			G2_bb += offs[4]
	#Get error vecs (residuals HT-G)
		T2aaerror, T2aaErr_vec = CCDutils.get_Err(ham.F_a,G2_aa,T2_aa,ham.nocca,ham.nvirta)
		T2aberror, T2abErr_vec = UCCSDutils.get_Err(ham.F_a,ham.F_b,G2_ab,T2_ab,ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb)
		T2bberror, T2bbErr_vec = CCDutils.get_Err(ham.F_b,G2_bb,T2_bb,ham.noccb,ham.nvirtb)
		T1aerror, T1aErr_vec = get_singles_Err(ham.F_a,G1_a,T1_a,ham.nocca,ham.nvirta)
		T1berror, T1bErr_vec = get_singles_Err(ham.F_b,G1_b,T1_b,ham.noccb,ham.nvirtb)
#		error = max(T2error,T1error)


   	#solve HT = G, damping amplitudes to improve convergence
		T1_a = solveccs(ham.F_a,G1_a,T1_a,ham.nocca,ham.nvirta,x=damping)
		T1_b = solveccs(ham.F_b,G1_b,T1_b,ham.noccb,ham.nvirtb,x=damping)
		T2_aa = CCDutils.solveccd(ham.F_a,G2_aa,T2_aa,ham.nocca,ham.nvirta,x=damping)
		T2_ab = UCCSDutils.solveccd(ham.F_a,ham.F_b,G2_ab,T2_ab,ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb,x=damping)
		T2_bb = CCDutils.solveccd(ham.F_b,G2_bb,T2_bb,ham.noccb,ham.nvirtb,x=damping)

	#Get convergence error
#		T2error = np.amax(T2-T2s[-1,:,:,:,:])
#		T1error = np.amax(T1-T1s[-1,:,:])
#		error = max(T2error,T1error)

   	#get energies
		ecorr = UCCSDutils.Ecorr(ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.nocca,ham.noccb) 
		error = abs(ecorr-eold)
#		error = 0
		print("Iteration = ", niter, " ECorr = ", ecorr, "error = ", error)
		eold = ecorr 
		niter += 1  
#

	if ((ampfile != 'none')):
		with open(ampfile, 'wb') as f:
			pickle.dump(T2_aa,f)
			pickle.dump(T2_ab,f)
			pickle.dump(T2_bb,f)
			pickle.dump(T1_a,f)
			pickle.dump(T1_b,f)
#
	ham.ecorr = ecorr
#	ham.T2 = np.copy(T2)
#	ham.T1 = np.copy(T1)
	
	
