import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import _thread

from pycuda.compiler import SourceModule

from matrix import *

mod=SourceModule("""
__global__ void calc(float *volume,float *input_matrix,float *output_volume)

{
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  
  int x=gridDim.x ; 
  int y=gridDim.y ;
  int p = i + j*x + k*x*y ; 
  
  int radius=int(x/2) ;
  int xin=i-radius ;
  int yin=j-radius ;
  int zin=k-radius ;
    
  float coord[3] ;
  int ti ;
  for (ti=0;ti<3;ti++)

  { 
    coord[ti]=radius+ xin*input_matrix[ti]+yin*input_matrix[3+ti]+zin*input_matrix[6+ti] ;
  }
    
  if ( abs(int(coord[0])) < 2*radius && abs(int(coord[1]))<2*radius && abs(int(coord[2]))<2*radius )
  {
    int ox=int(coord[0]);
    int oy=int(coord[1]);
    int oz=int(coord[2]);

    float dx=coord[0]-ox ;
    float dy=coord[1]-oy ;
    float dz=coord[2]-oz ;
     
    int ex=ox+1 ;
    int ey=oy+1 ;
    int ez=oz+1 ;
  
  float  value=(volume[ox + oy * x  + oz * x * y ]*(1-dx)*(1-dy)*(1-dz)+
         volume[ex + oy*x + oz*x*y]*dx*(1-dy)*(1-dz)+
         volume[ox+x*ey+x*y*oz]*(1-dx)*dy*(1-dz)+
         volume[ox+x*oy+x*y*ez]*(1-dx)*(1-dy)*dz+
         volume[ex+x*oy+x*y*ez]*(dx)*(1-dy)*dz+
         volume[ox+x*ey+x*y*ez]*(1-dx)*(dy)*dz+
         volume[ex+x*ey+x*y*oz]*(dx)*(dy)*(1-dz)+
         volume[ex+x*ey+x*y*ez]*(dx)*(dy)*dz )  ;

    output_volume[ p ] =value ;
  }
}

""")

func=mod.get_function("calc")



def gpu_rotate(a_gpu=None,R_gpu=None,b_gpu=None,grid=None):
  func(a_gpu,R_gpu,b_gpu,block=(1,1,1),grid=grid)
    




