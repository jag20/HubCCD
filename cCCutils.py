import numpy as np

def solveccd(F,G,T,nocc,nvirt,x=4.0):
  Tnew = np.zeros(np.shape(T),dtype=np.complex)
  for i in range(nocc):
    for j in range(nocc):
      for a in range(nvirt):
        aa = a + nocc
        for b in range(nvirt):
          bb = b + nocc
          d = (F[i,i] + F[j,j] - F[aa,aa] - F[bb,bb])
          Tnew[i,j,a,b] = G[i,j,a,b]/d
  #Damp amplitudes to improve convergence
  return(Tnew/x + T*(x-1.0)/x)


def solveccs(F,G1,T1,nocc,nvirt,x=4.0):
	#solve singles amplitude equations
	T1new = np.zeros(np.shape(T1),dtype=np.complex)
	for i in range(nocc):
		for a in range(nvirt):
			aa = a + nocc
			d = (F[i,i] - F[aa,aa])
			T1new[i,a] = G1[i,a]/d
	#Damp amplitudes to improve convergence
	return(T1new/x + T1*(x-1.0)/x)

