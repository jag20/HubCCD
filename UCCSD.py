from CCSDutils import *
#from UCCSDG2 import getccsdg2
from UCCSDG2 import getccsdg2, getg2, ccsdenergy
from UCCSDG1 import getg1
import UCCSDutils
import pickle
import CCD

def ccsd(ham,ampfile="none",variant="ccd"):

#read amplitudes from file if present to improve convergence
#	if ((ampfile != 'none') and(os.path.isfile(ampfile))):
#		with open(ampfile, 'rb') as f:
#			T2 = pickle.load(f)
#			T1 = pickle.load(f)
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
	eold  = 0.0e0
	#antisymm

	#Set up for CCD iteration and DIIS. interpolate doubles only for now
#	diis_start, diis_dim, Errors, T2s, Err_vec = CCDutils.diis_setup(ham.nocc,ham.nvirt)
#	T1Errors, T1s, T1Err_vec = diis_singles_setup(ham.nocc,ham.nvirt,diis_start,diis_dim)
#	G1_a, G1_b  = getg1(T1_a,T1_b,T2_aa,T2_ab,T2_bb,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca,ham.noccb,ham.nbas)
#	G2_aa, G2_ab, G2_bb = getccsdg2(T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca+1,ham.noccb+1)
	G2_aa, G2_ab, G2_bb = getg2(T2_aa,T2_ab,T2_bb,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca,ham.noccb,ham.nocca+1,ham.noccb+1,ham.nbas)
#g2aa,g2ab,g2bb = getccsdg2(t2aa,t2ab,t2bb,t1a,t1b,focka,fockb,eriaa,eriab,eribb,va,vb,[oa,ob,nbf])


	niter = 1
	tol = 1.0e-8
	error = tol*50
	damping= 1


	print("Beginning CCSD iteration")
	while (error > tol):
#		T2, Errors, T2s   = CCDutils.diis(diis_start,diis_dim,niter,Errors,T2s,T2,Err_vec)
#		T1, T1Errors, T1s = diis_singles(diis_start,diis_dim,niter,T1Errors,T1s,T1,T1Err_vec)
   	#build RHS G
#		G1_a, G1_b  = getg1(T1_a,T1_b,T2_aa,T2_ab,T2_bb,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb, ham.nocca,ham.noccb,ham.nbas)
#		G2_aa, G2_ab, G2_bb = getccsdg2(T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca,ham.noccb,ham.nbas)

#		G2_aa, G2_ab, G2_bb = getccsdg2(T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca+1,ham.noccb+1)
		G2_aa, G2_ab, G2_bb = getg2(T2_aa,T2_ab,T2_bb,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca,ham.noccb,ham.nocca+1,ham.noccb+1,ham.nbas)
#	#Get error vecs (residuals HT-G)
#		T1error, T1Err_vec = get_singles_Err(ham.F,G1,T1,ham.nocc,ham.nvirt)
#		error = max(T2error,T1error)

   	#solve HT = G
#		T1_a = solveccs(ham.F_a,G1_a,T1_a,ham.nocca,ham.nvirta,x=damping)
#		T1_b = solveccs(ham.F_b,G1_b,T1_b,ham.noccb,ham.nvirtb,x=damping)
#		T2_aa =   CCDutils.solveccd(ham.F_a,G2_aa,T2_aa,ham.nocca,ham.nvirta,x=damping)
#		T2_bb =   CCDutils.solveccd(ham.F_b,G2_bb,T2_bb,ham.noccb,ham.nvirtb,x=damping)
#		T2_ab = UCCSDutils.solveccd(ham.F_a,ham.F_b,G2_ab,T2_ab,ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb,x=damping)
##MP2 
#		T1_a = solveccs(ham.F_a,ham.F_a,T1_a,ham.nocca,ham.nvirta,x=damping)
#		T1_b = solveccs(ham.F_b,ham.F_b,T1_b,ham.noccb,ham.nvirtb,x=damping)
		T2_aa =   CCDutils.solveccd(ham.F_a,G2_aa,T2_aa,ham.nocca,ham.nvirta,x=damping)
		T2_bb =   CCDutils.solveccd(ham.F_b,G2_ab,T2_bb,ham.noccb,ham.nvirtb,x=damping)
#		G2ab = np.copy(ham.Eri_ab[ham.nocca:,ham.noccb:,:ham.nocca,:ham.noccb])
#		T2_ab = UCCSDutils.solveccd(ham.F_a,ham.F_b,G2ab,T2_ab,ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb,x=damping)
		T2_ab = UCCSDutils.solveccd(ham.F_a,ham.F_b,G2_ab,T2_ab,ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb,x=damping)
		print("T2 max")
		print(np.amax(T2_ab))
  #Damp amplitudes to improve convergence

	#Get convergence error
#		T2error = np.amax(T2-T2s[-1,:,:,:,:])
#		T1error = np.amax(T1-T1s[-1,:,:])
#		error = max(T2error,T1error)

   	#get energies
#		E1 = GCCSEn(ham.F,ham.Eri,T1,ham.nocc)
		#E2 = CCDutils.GCCDEn(ham.Eri_aa,T2_aa,ham.nocca)
		ecorr = UCCSDutils.Ecorr(ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.nocca,ham.noccb) #		ecorr = ccsdenergy(T1_a, T1_b, T2_aa, T2_ab, T2_bb, ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_bb,ham.Eri_ab,ham.nocca+1,ham.noccb+1,ham.nbas) 

#		ecorr = E1 + E2
		error = abs(ecorr-eold)
#		error = 0
		print("Iteration = ", niter, " ECorr = ", ecorr, "error = ", error)
		eold = ecorr 
		niter += 1  
#
		print("Error = ", error)

#	if ((ampfile != 'none')):
#		with open(ampfile, 'wb') as f:
#			pickle.dump(T2,f)
#			pickle.dump(T1,f)
#
	ham.ecorr = ecorr
#	ham.T2 = np.copy(T2)
#	ham.T1 = np.copy(T1)
	
	
