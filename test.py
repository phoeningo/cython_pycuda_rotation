print('import kernels...')
import mrcfile
from lib import method as M
#import rotate_core as C
#from rotate_core import *

import numpy as np
#from ftrotate import *
#from rslice import *
from pyrotate import *
from matrix import *
import pycuda.driver as cuda
from ftslice import *
import argparse
parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_mrc',type=str,default='test.mrc')
parser.add_argument('--output_mrc',type=str,default='tmp_rotate.mrc')
parser.add_argument('--angpix',type=float,default=1.6866)
parser.add_argument('--rot_seq',type=str,default='zxz')
parser.add_argument('--angle',type=str,default='0,0,0')
args=parser.parse_args()
print('done.')


angle=[]
ang_sp=args.angle.split(',')
for ang in ang_sp:
  angle.append(ang)

seq=args.rot_seq

input_volume,angpix=M.read_pix_mrc(args.input_mrc)
output_volume=np.empty_like(input_volume)
R=np.float32(euler2matrix(angle,seq))

#-----------------------

print('malloc device memory...')

total_mem=cuda.mem_alloc(input_volume.nbytes+output_volume.nbytes+R.nbytes)

a_gpu=total_mem
b_gpu=int(a_gpu)+input_volume.nbytes 
R_gpu=int(b_gpu)+output_volume.nbytes
print('done.')

cuda.memcpy_htod(a_gpu,input_volume)
cuda.memcpy_htod(R_gpu,R)


grid=input_volume.shape

gpu_rotate(np.intp(a_gpu),np.intp(R_gpu),np.intp(b_gpu),grid)

cuda.memcpy_dtoh(output_volume,b_gpu)
#M.write_file(np.float32(fo_volume),'test_slice.mrc')
#output_volume=np.float32(np.abs( np.fft.ifftn ( np.fft.ifftshift(fo_volume))))
#output_volume=np.float32(np.abs( np.fft.ifftn ( fo_volume)))

M.write_pix_file(zslice(output_volume),args.output_mrc,angpix)



