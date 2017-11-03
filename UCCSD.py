import numpy as np
import os
import pickle
from CCSDutils import *
from UCCSDG2 import getccsdg2, getg2, ccsdenergy
from UCCSDG1 import getg1
import UCCSDutils
import CCD
import CCDutils
#This function implements spin-summed UCCSD that can be used with UHF and ROHF references. RHF reference#is fine too, but we have to convert it to the UHF basis.
#We also implement ROCCSD0 (singlet-paired coupled cluster) from John Gomez, Tom Henderson and Gus 
#Scuseria, JCP 144, 244117 (2016).

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

	variant = variant.lower()

	#Set up a few things for singlet-paired coupled cluster
	ham.F_a = np.copy(ham.F_a)
	ham.F_b = np.copy(ham.F_b)
	s_o = ham.nocca-ham.noccb 
	if (variant == 'ccsd0'):
		if (ham.wfn_type == 'rhf'):
			variant = 'rccsd0'
		elif (ham.wfn_type == 'uhf'):
			variant = 'roccsd0'
   
#read amplitudes from file if present to improve convergence
	if ((ampfile != 'none') and(os.path.isfile(ampfile))):
		with open(ampfile, 'rb') as f:
			T2_aa = pickle.load(f)
			T2_ab = pickle.load(f)
			T2_bb = pickle.load(f)
			T1_a = pickle.load(f)
			T1_b = pickle.load(f)

	else:
		T2_aa = np.zeros([ham.nocca,ham.nocca,ham.nvirta,ham.nvirta],order='F')
		T2_ab = np.zeros([ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb],order='F')
		T2_bb = np.zeros([ham.noccb,ham.noccb,ham.nvirtb,ham.nvirtb],order='F')
		T1_a  = np.zeros([ham.nocca,ham.nvirta],order='F')
		T1_b  = np.zeros([ham.noccb,ham.nvirtb],order='F')

	#useful to initialize RHS arrays in Fortran order, since coupled cluster RHS will be built in Fortran
	G2_aa = np.zeros([ham.nocca,ham.nocca,ham.nvirta,ham.nvirta],order='F')
	G2_ab = np.zeros([ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb],order='F')
	G2_bb = np.zeros([ham.noccb,ham.noccb,ham.nvirtb,ham.nvirtb],order='F')
	G1_a  = np.zeros([ham.nocca,ham.nvirta],order='F')
	G1_b  = np.zeros([ham.noccb,ham.nvirtb],order='F')

	#Set up for CCSD iteration and DIIS. 
	diis_start, diis_dim, T2aaErrors, T2aas, T2aaErr_vec = CCDutils.diis_setup(ham.nocca,ham.nvirta)
	T2abErrors, T2abs, T2abErr_vec = UCCSDutils.diis_setup(diis_start,diis_dim,ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb)
	T2bbErrors, T2bbs, T2bbErr_vec = UCCSDutils.diis_setup(diis_start,diis_dim,ham.noccb,ham.noccb,ham.nvirtb,ham.nvirtb)
	T1aErrors, T1as, T1aErr_vec = diis_singles_setup(ham.nocca,ham.nvirta,diis_start,diis_dim)
	T1bErrors, T1bs, T1bErr_vec = diis_singles_setup(ham.noccb,ham.nvirtb,diis_start,diis_dim)

	niter = 1
	tol = 1.0e-8
	error = tol*50
	damping= 2
	eold  = 0.0e0
#
#	#Check offdiagonal terms to see if we're in a non-canonical basis
	F_a_offdiag = np.copy(ham.F_a)
	F_b_offdiag = np.copy(ham.F_b)
	np.fill_diagonal(F_a_offdiag,0.0)
	np.fill_diagonal(F_b_offdiag,0.0)
	if (np.amax(F_a_offdiag[:ham.nocca,:ham.nocca] > tol) or np.amax(F_b_offdiag[:ham.noccb,:ham.noccb]) > tol) :
		print("Using noncanonical basis")
		NObas = True
	else:
		NObas = False 


	#Need this for fortran routines
	va  = ham.nocca + 1
	vb  = ham.noccb + 1



	print("Beginning UCCSD iteration")
	while (error > tol):
		#Extrapolate amplitudes using DIIIS
		T2_aa, T2aaErrors, T2aas   = CCDutils.diis(diis_start,diis_dim,niter,T2aaErrors,T2aas,T2_aa,T2aaErr_vec)
		T2_ab, T2abErrors, T2abs   = CCDutils.diis(diis_start,diis_dim,niter,T2abErrors,T2abs,T2_ab,T2abErr_vec)
		T2_bb, T2bbErrors, T2bbs   = CCDutils.diis(diis_start,diis_dim,niter,T2bbErrors,T2bbs,T2_bb,T2bbErr_vec)
		T1_a, T1aErrors, T1as = diis_singles(diis_start,diis_dim,niter,T1aErrors,T1as,T1_a,T1aErr_vec)
		T1_b, T1bErrors, T1bs = diis_singles(diis_start,diis_dim,niter,T1bErrors,T1bs,T1_b,T1bErr_vec)

			#Symmetrize oo-vv block again
		T2_aa[:ham.noccb,:ham.noccb,:,:] = 0.50e0*(T2_aa[:ham.noccb,:ham.noccb,:,:]+ np.swapaxes(T2_aa[:ham.noccb,:ham.noccb,:,:],2,3))
		T2_ab[:ham.noccb,:,:,s_o:] = 0.50e0*(T2_ab[:ham.noccb,:,:,s_o:]+ np.swapaxes(T2_ab[:ham.noccb,:,:,s_o:],2,3))
		T2_bb[:ham.noccb,:ham.noccb,s_o:,s_o:] = 0.50e0*(T2_bb[:,:,s_o:,s_o:]+ np.swapaxes(T2_bb[:,:,s_o:,s_o:],2,3))
		


	   	#build RHS G
		G1_a, G1_b  = getg1(T1_a,T1_b,T2_aa,T2_ab,T2_bb,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb, ham.nocca,ham.noccb,ham.nbas)
		G2_aa, G2_ab, G2_bb = getccsdg2(T2_aa,T2_ab,T2_bb,T1_a,T1_b,ham.F_a,ham.F_b,ham.Eri_aa,ham.Eri_ab,ham.Eri_bb,ham.nocca,ham.noccb,ham.nbas)

#
		#Symmetrize G for CCSD0
		if (variant == 'rccsd0'):
			G2_aa = 0.50e0*(G2_aa + np.swapaxes(G2_aa,2,3))
			G2_ab = 0.50e0*(G2_ab + np.swapaxes(G2_ab,2,3))
			G2_bb = 0.50e0*(G2_bb + np.swapaxes(G2_bb,2,3))
		elif (variant == 'roccsd0'):
			#Symmetrize oo-vv block
			G2_aa[:ham.noccb,:ham.noccb,:,:] = 0.50e0*(G2_aa[:ham.noccb,:ham.noccb,:,:]+ 
											  np.swapaxes(G2_aa[:ham.noccb,:ham.noccb,:,:],2,3))
			G2_ab[:ham.noccb,:,:,s_o:] = 0.50e0*(G2_ab[:ham.noccb,:,:,s_o:]+ np.swapaxes(G2_ab[:ham.noccb,:,:,s_o:],2,3))
			G2_bb[:ham.noccb,:ham.noccb,s_o:,s_o:] = 0.50e0*(G2_bb[:,:,s_o:,s_o:]+ np.swapaxes(G2_bb[:,:,s_o:,s_o:],2,3))

			#Then move open-shell coupling terms
			G2_aa[:ham.noccb,:ham.noccb,:,:] -=(np.einsum('ix,xjab->ijab',
          			 ham.F_a[:ham.noccb,ham.noccb:ham.nocca],T2_aa[ham.noccb:ham.nocca,:ham.noccb,:,:])
			       + np.einsum('jx,ixab->ijab',
		             ham.F_a[:ham.noccb,ham.noccb:ham.nocca],T2_aa[:ham.noccb,ham.noccb:ham.nocca,:,:]))
			G2_bb[:,:,s_o:,s_o:] +=(np.einsum('ax,ijxb->ijab',
		     		 ham.F_b[ham.nocca:,ham.noccb:ham.nocca],T2_bb[:,:,:s_o,s_o:])
			       + np.einsum('bx,ijax->ijab',
	     			 ham.F_b[ham.nocca:,ham.noccb:ham.nocca],T2_bb[:,:,s_o:,:s_o]))
			G2_ab[:ham.noccb,:,:,s_o:] +=(-np.einsum('ix,xjab->ijab',
          			 ham.F_a[:ham.noccb,ham.noccb:ham.nocca],T2_ab[ham.noccb:ham.nocca,:,:,s_o:])
			       + np.einsum('bx,ijax->ijab',
	     			 ham.F_b[ham.nocca:,ham.noccb:ham.nocca],T2_ab[:ham.noccb,:,:,:s_o]))
			#Symmetrize oo-vv block again
			G2_aa[:ham.noccb,:ham.noccb,:,:] = 0.50e0*(G2_aa[:ham.noccb,:ham.noccb,:,:]+ np.swapaxes(G2_aa[:ham.noccb,:ham.noccb,:,:],2,3))
			G2_ab[:ham.noccb,:,:,s_o:] = 0.50e0*(G2_ab[:ham.noccb,:,:,s_o:]+ np.swapaxes(G2_ab[:ham.noccb,:,:,s_o:],2,3))
			G2_bb[:ham.noccb,:ham.noccb,s_o:,s_o:] = 0.50e0*(G2_bb[:,:,s_o:,s_o:]+ np.swapaxes(G2_bb[:,:,s_o:,s_o:],2,3))
		
   	#solve HT = G, damping amplitudes to improve convergence.
		

		if (NObas): #Use conjugate gradients if we're not in a canonical basis
			T1_anew = UCCSDutils.SolveT1_CG(ham.F_a,T1_a,G1_a,ham.nocca,ham.nvirta)
			T1_a = (T1_anew/damping + T1_a*(damping-1.0)/damping)
			T1_bnew = UCCSDutils.SolveT1_CG(ham.F_b,T1_b,G1_b,ham.noccb,ham.nvirtb)
			T1_b = (T1_bnew/damping + T1_b*(damping-1.0)/damping)
			T2_aa_new = UCCSDutils.SolveT2_CG(ham.F_a,ham.F_a,T2_aa,G2_aa,ham.nocca,ham.nocca,ham.noccb,ham.nocca,variant)
			T2_ab_new = UCCSDutils.SolveT2_CG(ham.F_a,ham.F_b,T2_ab,G2_ab,ham.nocca,ham.noccb,ham.noccb,ham.nocca,variant)
			T2_bb_new = UCCSDutils.SolveT2_CG(ham.F_b,ham.F_b,T2_bb,G2_bb,ham.noccb,ham.noccb,ham.noccb,ham.nocca,variant)
			T2_aa = (T2_aa_new/damping + T2_aa*(damping-1.0)/damping)
			T2_ab = (T2_ab_new/damping + T2_ab*(damping-1.0)/damping)
			T2_bb = (T2_bb_new/damping + T2_bb*(damping-1.0)/damping)

		else:
			T1_a = solveccs(ham.F_a,G1_a,T1_a,ham.nocca,ham.nvirta,x=damping)
			T1_b = solveccs(ham.F_b,G1_b,T1_b,ham.noccb,ham.nvirtb,x=damping)
			T2_aa =   CCDutils.solveccd(ham.F_a,G2_aa,T2_aa,ham.nocca,ham.nvirta,x=damping)
			T2_ab = UCCSDutils.solveccd(ham.F_a,ham.F_b,G2_ab,T2_ab,ham.nocca,ham.noccb,ham.nvirta,ham.nvirtb,x=damping)
			T2_bb =   CCDutils.solveccd(ham.F_b,G2_bb,T2_bb,ham.noccb,ham.nvirtb,x=damping)


		if (variant == 'rccsd0'):
			T2_aa = 0.50e0*(T2_aa + np.swapaxes(T2_aa,2,3))
			T2_ab = 0.50e0*(T2_ab + np.swapaxes(T2_ab,2,3))
			T2_bb = 0.50e0*(T2_bb + np.swapaxes(T2_bb,2,3))

		elif (variant == 'roccsd0'):
			#Symmetrize oo-vv block
			T2_aa[:ham.noccb,:ham.noccb,:,:] = 0.50e0*(T2_aa[:ham.noccb,:ham.noccb,:,:]+ 
											   np.swapaxes(T2_aa[:ham.noccb,:ham.noccb,:,:],2,3))
			T2_ab[:ham.noccb,:,:,s_o:] = 0.50e0*(T2_ab[:ham.noccb,:,:,s_o:]+ np.swapaxes(T2_ab[:ham.noccb,:,:,s_o:],2,3))
			T2_bb[:ham.noccb,:ham.noccb,s_o:,s_o:] = 0.50e0*(T2_bb[:,:,s_o:,s_o:]+ np.swapaxes(T2_bb[:,:,s_o:,s_o:],2,3))

	#Get error vecs (residuals HT-G)
		T2aaerror, T2aaErr_vec = UCCSDutils.get_Err(ham.F_a,ham.F_a,G2_aa,T2_aa,ham.nocca,ham.nocca,ham.noccb,ham.nocca,variant)
		T2aberror, T2abErr_vec = UCCSDutils.get_Err(ham.F_a,ham.F_b,G2_ab,T2_ab,ham.nocca,ham.noccb,ham.noccb,ham.nocca,variant)
		T2bberror, T2bbErr_vec = UCCSDutils.get_Err(ham.F_b,ham.F_b,G2_bb,T2_bb,ham.noccb,ham.noccb,ham.noccb,ham.nocca,variant)
		T1aerror, T1aErr_vec   = UCCSDutils.get_singles_Err(ham.F_a,G1_a,T1_a,ham.nocca,ham.nvirta)
		T1berror, T1bErr_vec   = UCCSDutils.get_singles_Err(ham.F_b,G1_b,T1_b,ham.noccb,ham.nvirtb)
		if (variant == 'roccsd0'):
			#Symmetrize oo-vv block
			T2aaErr_vec[:ham.noccb,:ham.noccb,:,:] = 0.50e0*(T2aaErr_vec[:ham.noccb,:ham.noccb,:,:]+ 
											   np.swapaxes(T2aaErr_vec[:ham.noccb,:ham.noccb,:,:],2,3))
			T2abErr_vec[:ham.noccb,:,:,s_o:] = 0.50e0*(T2abErr_vec[:ham.noccb,:,:,s_o:]+ np.swapaxes(T2abErr_vec[:ham.noccb,:,:,s_o:],2,3))
			T2bbErr_vec[:ham.noccb,:ham.noccb,s_o:,s_o:] = 0.50e0*(T2bbErr_vec[:,:,s_o:,s_o:]+ np.swapaxes(T2bbErr_vec[:,:,s_o:,s_o:],2,3))
#		error = max(T2aaerror,T2aberror,T2bberror,T1aerror,T1berror)


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
	
	
