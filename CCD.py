import numpy as np
import os 
from CCDutils import *

#This module actually runs the ccd iteration, calling the necessary functions from CCDutils

def ccd(ham,ampfile="none",variant="ccd"):
  import pickle
  import os
  Gs  = {"ghf":GHFCCD, "rhf": RHFCCD}
  ens = {"ghf":GCCDEn, "rhf": RCCDEn}


  getG   = Gs[ham.wfn_type]
  energy = ens[ham.wfn_type]
   

  damping=1
  #read amplitudes from file if present to improve convergence
  if ((ampfile != 'none') and(os.path.isfile(ampfile))):
    with open(ampfile, 'rb') as f:
      T = pickle.load(f)
      ecorr = energy(ham.Eri,T,ham.nocc)
      eold = ecorr
  else:
    T = np.zeros([ham.nocc,ham.nocc,ham.nvirt,ham.nvirt])
    eold = 0.0e0

  Told = np.copy(T)
  tol = 1.0e-09

  #Set up for CCD iteration and DIIS
  diis_start, diis_dim, Errors, Ts, Err_vec = diis_setup(ham.nocc,ham.nvirt,np.float)
  niter = 1
  error = tol*50

  #Do CCD
  while (error > tol):
    #extrapolate amplitudes using DIIS
    T, Errors, Ts = diis(diis_start,diis_dim,niter,Errors,Ts,T,Err_vec)
    G = getG(ham.F,ham.Eri,T,ham.nocc,ham.nbas,niter,variant=variant)
    error, Err_vec = get_Err(ham.F,G,T,ham.nocc,ham.nvirt)

    T = solveccd(ham.F,G,T,ham.nocc,ham.nvirt,x=damping)
    ecorr = energy(ham.Eri,T,ham.nocc)
    
    niter += 1

	#Get convergence error
    error = np.amax(T-Ts[-1,:,:,:,:])

    print("Energy = ", ecorr, "error = ", error)
    ham.ecorr = ecorr
    ham.T2 = T

  

  #save amplitudes if requested
  if (ampfile.lower() != "none"):
    with open(ampfile, "wb") as f:
      pickle.dump(T, f)
 
