import numpy as np
#Here we define some extra utilities needed for spin-summed, unrestricted CCSD

def solveccd(F_a,F_b,G,T,nocca,noccb,nvirta,nvirtb,x=4.0):
  #Solve for opposite-spin amplitudes
  Tnew = np.zeros(np.shape(T))
  for i in range(nocca):
    for j in range(noccb):
      for a in range(nvirta):
        aa = a + nocca
        for b in range(nvirtb):
          bb = b + noccb
          d = (F_a[i,i] + F_b[j,j] - F_a[aa,aa] - F_b[bb,bb])
          Tnew[i,j,a,b] = G[i,j,a,b]/d
  #Damp amplitudes to improve convergence
  print("x = ", x)
  return(Tnew/x + T*(x-1.0)/x)

def Ecorr(F_a,F_b,Eri_aa,Eri_ab,Eri_bb,T2_aa,T2_ab,T2_bb,T1_a,T1_b,nocca,noccb):
    #unrestricted, spin-summation of the CCSD correlation energy
    #CCD piece
	Eaa  = 0.25e0*np.einsum('ijab,abij',T2_aa,Eri_aa[nocca:,nocca:,:nocca,:nocca])
	Ebb = 0.25e0*np.einsum('ijab,abij',T2_bb,Eri_bb[noccb:,noccb:,:noccb,:noccb])
	Eab = np.einsum('ijab,abij',T2_ab,Eri_ab[nocca:,noccb:,:nocca,:noccb])
	print("aa = ", Eaa)
	print("bb = ", Ebb)
	print("ab = ", Eab)
	E2 = Eaa + Ebb + Eab
	#linear in singles
	E1 = np.einsum('ia,ai',T1_a,F_a[nocca:,:nocca])
	E1 += np.einsum('ia,ai',T1_b,F_b[noccb:,:noccb])
    
	#quadratic in singles
	E1 += np.einsum('ia,ai',T1_a,F_a[nocca:,:nocca])
	E1 += 0.5e0*np.einsum('ia,jb,abij',T1_a,T1_a,Eri_aa[nocca:,nocca:,:nocca,:nocca])
	E1 += 0.5e0*np.einsum('ia,jb,abij',T1_a,T1_b,Eri_ab[nocca:,noccb:,:nocca,:noccb])
	E1 += 0.5e0*np.einsum('ia,jb,baji',T1_b,T1_a,Eri_ab[nocca:,noccb:,:nocca,:noccb])
	E1 += 0.5e0*np.einsum('ia,jb,abij',T1_b,T1_b,Eri_bb[noccb:,noccb:,:noccb,:noccb])
	print(np.amax(T1_a))
	print(np.amax(T1_b))
	print("E1 = ", E1)
	return E1+E2
