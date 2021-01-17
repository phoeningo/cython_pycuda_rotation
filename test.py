print('import kernels...')
import mrcfile
from lib import method as M
#import rotate_core as C
#from rotate_core import *

import numpy as np
from pyrotate import *
from matrix import *
import pycuda.driver as cuda

import argparse
parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_mrc',type=str,default='test.mrc')
parser.add_argument('--output_mrc',type=str,default='tmp_rotate.mrc')
parser.add_argument('--angpix',type=float,default=1.6866)
parser.add_argument('--rot_seq',type=str,default='zxz')
parser.add_argument('--angle',type=str,default='90,0,0')
args=parser.parse_args()
print('done.')


angle=[]
ang_sp=args.angle.split(',')
for ang in ang_sp:
  angle.append(ang)

seq=args.rot_seq

#print('reading input volume...')
input_volume,angpix=M.read_pix_mrc(args.input_mrc)
#print('done.')
#print('malloc host memory...')
output_volume=np.empty_like(input_volume)
R=np.float32(invMatrix(angle,seq))

print('malloc device memory...')
a_gpu=cuda.mem_alloc(input_volume.nbytes)
b_gpu=cuda.mem_alloc(input_volume.nbytes)
R_gpu=cuda.mem_alloc(R.nbytes)

print('done.')
cuda.memcpy_htod(a_gpu,input_volume)
cuda.memcpy_htod(b_gpu,input_volume)
cuda.memcpy_htod(R_gpu,R)


grid=input_volume.shape
gpu_rotate(a_gpu,R_gpu,b_gpu,grid)

cuda.memcpy_dtoh(output_volume,b_gpu)
#cuda.free()

#cuda_test(input_volume,R,output_volume)
#volume=C.rotate(input_volume,angle,seq)
M.write_pix_file(output_volume,args.output_mrc,angpix)



