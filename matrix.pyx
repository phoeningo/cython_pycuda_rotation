
import math
import numpy as np

import time

def Rx(input_angle):
  theta=np.float32(input_angle)
  TT=np.matrix([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
  #print(TT)
  return TT

def Ry(input_angle):
  phi=np.float32(input_angle)
  TT=np.matrix([ [math.cos(phi),0,math.sin(phi)] , [0,1,0] , [-math.sin(phi),0,math.cos(phi)] ] )
 # print( TT)
  return TT


def Rz(input_angle):
  psi=np.float32(input_angle)
  TT= np.matrix([[math.cos(psi),-math.sin(psi),0],[math.sin(psi),math.cos(psi),0],[0,0,1]])
  #print (TT)
  return TT

  
def euler2matrix(input_angle,rotate_seq):
  #phi,theta,psi=input_angle
  #angles=[]  
  Rs=[]
  for ri in range(3):
    if rotate_seq[ri]=='x':
      Rs.append(Rx(input_angle[ri]))
    if rotate_seq[ri]=='y':
      Rs.append(Ry(input_angle[ri]))
    if rotate_seq[ri]=='z':
      Rs.append(Rz(input_angle[ri]))

  return np.array((Rs[0]*Rs[1]*Rs[2]))

def invMatrix(input_angle,rotate_seq):
  #phi,theta,psi=input_angle
  #angles=[]
  Rs=[]
  for ri in range(3):
    if rotate_seq[ri]=='x':
      Rs.append(Rx(input_angle[ri]).T)
    if rotate_seq[ri]=='y':
      Rs.append(Ry(input_angle[ri]).T)
    if rotate_seq[ri]=='z':
      Rs.append(Rz(input_angle[ri]).T)

  return np.array((Rs[2]*Rs[1]*Rs[0]))
