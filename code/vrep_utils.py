# Contains utility functions to send and recieve data with V-REP
import numpy as np
import vrep
import ctypes

class SendData( object ):

  def __init__( self, signal_name, size=None, cid=0 ):
    self.cid = cid
    self.mode = vrep.simx_opmode_oneshot
    self.signal_name = signal_name
    self.size = size
  
  def __call__( self, t, values ):
    if self.size is None:
      packedData=vrep.simxPackFloats(values.flatten())
    else:
      packedData=vrep.simxPackFloats(values[:self.size].flatten())
    raw_bytes = (ctypes.c_ubyte * len(packedData)).from_buffer_copy(packedData) 
    err = vrep.simxSetStringSignal(self.cid, self.signal_name,
                                    raw_bytes,
                                    self.mode)
