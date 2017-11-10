import sys
sys.path.append('../HubCCD')
import ham 
import numpy as np
import CCD
import CCSD
import UCCSD
from ast import literal_eval
import cGCCSD
from scf import twoe_MO_tran

fle = '16x1_complex_ghf'
#This function reads complex_GHF MOs and data from Kitou's Fortran output files
def read_K(fle):
	#Get header info
	with open(fle) as f:
		obj = f.read().splitlines()
		nsitesx = int(obj[1])
		nsitesy = 1
		PeriodicX = True
		nocc = int(obj[2])
		fill = nocc/(2.0*nsitesx)
		U = float(obj[3])
		
		
	ml = ham.hub(nsitesx=nsitesx,nsitesy=nsitesy,U=U,fill=fill,PeriodicX=PeriodicX)
	#build AO ints, don't really need them for fast GHF
	ml.get_ints()
	ml.nso =   2*ml.nbas
	ml.nocc  = 2*ml.nocc
	ml.nvirt = 2*ml.nvirt
	ml.wfn_type = 'ghf'
	#initialize arrays
	ml.C = np.zeros((ml.nso,ml.nso),dtype=np.complex)
	ml.F = np.zeros((ml.nso,ml.nso),dtype=np.complex)
	with open(fle) as f:
		obj = f.read().splitlines()
	c = [line.strip() for line in obj[4:(4+ml.nso**2)]]
	go_comp = lambda x: complex(literal_eval(x)[0],literal_eval(x)[1])
	c_nums = np.array([go_comp(num) for num in c])
	ml.C = np.reshape(c_nums,(ml.nso,ml.nso)).T
	#assume orbital energies are real
	f = [line.strip() for line in obj[(4+ml.nso**2):]]
	f_nums = np.loadtxt(f,dtype=np.complex)
	ml.F = np.diag(f_nums)
	return ml
