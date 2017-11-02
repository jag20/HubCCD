# This file interfaces to Gaussian using the Gaussian utilities qcmatrixio, QCOpMat and QCMatEl
# described in and available from http://gaussian.com/interfacing/
# Here, we read various integrals and scalars needed for post-HF calculations
# Note that this script and the underlying modules require
# python3.
#
import sys
import numpy as np
import qcmatrixio as qcio
import QCOpMat as qco
import QCMatEl as qcm

def read_ints(wfn_type,fname):
# fname is the name of a matrix element file computed by Gaussian.
  
# Create the container object with the contents of the
# specified file
  mel = qcm.MatEl(file=fname)
# Both the container object and the objects for each item
# within it have appropriate print methods and can be
# printed in the usual way.

# compute derived scalars from the values in the object
  nae = int(int(mel.ne+mel.multip-1)/2)  # number of alpha electrons
  nbe = int(int(mel.ne-mel.multip+1)/2)  # number of beta electrons
  noa = nae - mel.nfc                    # number of active (not frozen
  nob = nbe - mel.nfc                    # core or virtual) electrons
  nrorb = mel.nbsuse - mel.nfc - mel.nfv # number of active orbitals
  nva = nrorb - noa                      # number of active alpha virtuals
  nvb = nrorb - nob                      # number of active betavirtuals

  if mel.itran == 5:
    ldima = nrorb
    ldimb = nrorb
  else:
    ldima = noa
    ldimb = nob

  #Get necessary arrays and transpose, since Gaussian writes in Fortran style
  aoe  = mel.matlist["ALPHA ORBITAL ENERGIES"].expand().reshape((mel.nbsuse))
  F_a = mel.matlist["ALPHA FOCK MATRIX"].expand().reshape((nrorb,nrorb)).T
  C_a = mel.matlist["ALPHA MO COEFFICIENTS"].expand().reshape((nrorb,nrorb)).T
  P = mel.matlist["ALPHA SCF DENSITY MATRIX"].expand().reshape((nrorb,nrorb)).T #P is in AO basis
  X = mel.matlist["ORTHOGONAL BASIS"].expand().reshape((nrorb,nrorb)).T
  Xinv = np.linalg.inv(X)
  H = mel.matlist["CORE HAMILTONIAN ALPHA"].expand().reshape((nrorb,nrorb)).T
  S = mel.matlist["OVERLAP"].expand().reshape((nrorb,nrorb)).T
  aa2e = mel.matlist["AA MO 2E INTEGRALS"].expand().reshape((ldima,nrorb,nrorb,nrorb)).T

  #convert to orthogonalized MO basis, C will do this for us
  F_a = np.dot(C_a.T,np.dot(F_a,C_a))
  H_aa = np.dot(C_a.T,np.dot(H,C_a))
  #ERIs to Dirac ordering
  ERI_aa = np.swapaxes(aa2e,1,2)

  #other scalars
  eref = mel.scalar("escf")
  nrep = mel.scalar("enucrep")

  if (wfn_type == "rhf"):
    print("Returning RHF molecular parameters")
    return nrorb, noa, nva, eref, C_a, F_a, ERI_aa
  if ((wfn_type =='uhf') or (wfn_type == 'rohf')):
    F_b = mel.matlist["BETA FOCK MATRIX"].expand().reshape((nrorb,nrorb)).T
    C_b = mel.matlist["BETA MO COEFFICIENTS"].expand().reshape((nrorb,nrorb)).T
    F_b = np.dot(C_b.T,np.dot(F_b,C_b))
    ab2e = np.swapaxes(mel.matlist["BA MO 2E INTEGRALS"].expand().reshape((ldima,nrorb,nrorb,nrorb)).T,0,2)
    ab2e = np.swapaxes(ab2e,1,3)
    bb2e = mel.matlist["BB MO 2E INTEGRALS"].expand().reshape((ldima,nrorb,nrorb,nrorb)).T
    ERI_ab = np.swapaxes(ab2e,1,2)
    ERI_bb = np.swapaxes(bb2e,1,2)
    if (wfn_type == 'rohf'):
      P_a = mel.matlist["ALPHA SCF DENSITY MATRIX"].expand().reshape((nrorb,nrorb)).T #P is in AO basis
      P_b = mel.matlist["BETA SCF DENSITY MATRIX"].expand().reshape((nrorb,nrorb)).T #P is in AO basis
      return nrorb, noa, nob, nva, nvb, C_a, ERI_aa, P_a, P_b, X, S, H_aa, nrep
#    return nrorb, noa, nob, nva, nvb, eref, C_a, C_b, F_a, F_b, ERI_aa, ERI_ab, ERI_bb
    print("Returning UHF molecular parameters")
    return nrorb, noa, nob, nva, nvb, eref, C_a, C_b, F_a, F_b, ERI_aa, ERI_ab, ERI_bb
#    return nrorb, noa, noa, nva, nva, eref, C_a, C_a, F_a, F_a, ERI_aa, ERI_aa, ERI_aa
  
#  else:
#    print("Error reading Gaussian matrix element file")
#
