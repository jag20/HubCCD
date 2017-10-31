import numpy as np
from scf import moUHF_to_GHF
#This routine implements complex generalized CCSD for the Hubbard Hamiltonian in N^5 scaling
#by exploiting the sparsity of the two-electron integrals for Hubbard. Derivation and factorization
#of the spin-orbital CCSD equations were worked out by Tom Henderson.
def ccsd(ham,ampfile="none"):
	if (ham.hamtype != 'Hubbard'):
		print("This routine should only be used for Hubbard in the spin-orbital basis.")
	if (ham.wfn_type == 'uhf'):
		print("converting UHF wavefunction to spin-orbital basis")
		ham.F, ham.Eri, ham.C = moUHF_to_GHF(ham.C_a,ham.C_b,ham.F_a,ham.F_b,ham.Eri_aa,ham.nocca,ham.noccb,ham.nbas)
		ham.nso = 2*ham.nbas
		ham.nocc  = ham.nocca + ham.noccb
		ham.nvirt = ham.nvirta + ham.nvirtb
		ham.wfn_type = 'ghf'

	#Build some initial intermediates. ham.C is AOxMO, with alpha in the first nbas/2 rows, followed by beta.
	C_up   =  ham.C[:ham.nbas,:]
	C_down =  ham.C[ham.nbas:,:]

	Cuij = np.zeros((ham.nbas,ham.nso,ham.nso),dtype=np.complex)
	Cabu = np.zeros((ham.nso,ham.nso,ham.nbas),dtype=np.complex)
	for u in range(ham.nbas):
		for i in range(ham.nso):
			for j in range(ham.nso):
				Cuij[u,i,j] = C_up[u,i]*C_down[u,j] - C_down[u,i]*C_up[u,j] 
				Cabu[i,j,u] = (np.conj(C_up[u,i])*np.conj(C_down[u,j])
							- np.conj(C_down[u,i])*np.conj(C_up[u,j]))

	print("done")
