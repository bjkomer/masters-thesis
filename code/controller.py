# A Set of standard non-neural controllers to benchmark against
import numpy as np
import time
import vrep
from quadcopter import Quadcopter

class PD( Quadcopter ):

  def __init__( self, **kwargs ):

    super(PD, self).__init__( **kwargs );
    
    k1 = 0.43352026190263104
    k2 = 2.0 * 4
    k3 = 0.5388202808181405
    k4 = 1.65 * 4
    k5 = 2.5995452450850185
    k6 = 0.802872750102059 * 8
    k7 = 0.5990281657438163
    k8 = 2.8897310746350824 * 4

    self.gains = np.matrix([[ 0,  0, k2,  0,  0,-k4,  0,  0,  0,  0,  0,  0],
                            [  0, k1,  0,  0,-k3,  0,-k5,  0,  0, k7,  0,  0],
                            [-k1,  0,  0, k3,  0,  0,  0,-k5,  0,  0, k7,  0],
                            [  0,  0,  0,  0,  0,  0,  0,  0,-k6,  0,  0, k8] ])

    self.rotor_transform = np.matrix([[ 1,-1, 1, 1],
                                      [ 1,-1,-1,-1],
                                      [ 1, 1,-1, 1],
                                      [ 1, 1, 1,-1] ])

    self.state = np.matrix([[0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                           ])

    self.gravity_compensation = np.matrix([[5.6535],
                                           [5.6535],
                                           [5.6535],
                                           [5.6535],
                                          ])

  def control_step( self ):

    self.count += 1
    if self.count == 10:
      self.get_target()
      self.calculate_error()
      
      self.state = np.matrix([[self.pos_err[0]],
                         [self.pos_err[1]],
                         [self.pos_err[2]],
                         [self.lin[0]],
                         [self.lin[1]],
                         [self.lin[2]],
                         [self.ori_err[0]],
                         [self.ori_err[1]],
                         [self.ori_err[2]],
                         [self.ang[0]],
                         [self.ang[1]],
                         [self.ang[2]],
                        ])
      
      self.count = 0
      self.send_motor_commands( self.compute_output() )
      vrep.simxSynchronousTrigger( self.cid )

  def compute_output( self ):
    """ Computes the rotor velocities based on PID control """

    motor = self.rotor_transform * (self.gains * self.state) + self.gravity_compensation

    return [motor[0,0], motor[1,0], motor[2,0], motor[3,0]]

class PID( PD ):

  def __init__( self, **kwargs ):

    super(PID, self).__init__(**kwargs);

    i1 = 0.0001/10
    i2 = 0.05/10
    i3 = 0#0.0001
    i4 = 0#0.0001

    self.I_gain = np.matrix([[  0,  0, i2,  0,  0,  0],
                             [  0, i1,  0,-i3,  0,  0],
                             [-i1,  0,  0,  0,-i3,  0],
                             [  0,  0,  0,  0,  0,-i4],
                            ])

    self.integrals = np.matrix([[0.0], # X
                                [0.0], # Y
                                [0.0], # Z
                                [0.0], # Roll
                                [0.0], # Pitch
                                [0.0], # Yaw
                               ])

  def control_step( self ):

    self.count += 1
    if self.count == 10:
      self.get_target()
      self.calculate_error()
      
      self.state = np.matrix([[self.pos_err[0]],
                         [self.pos_err[1]],
                         [self.pos_err[2]],
                         [self.lin[0]],
                         [self.lin[1]],
                         [self.lin[2]],
                         [self.ori_err[0]],
                         [self.ori_err[1]],
                         [self.ori_err[2]],
                         [self.ang[0]],
                         [self.ang[1]],
                         [self.ang[2]],
                        ])
    
      self.update_integral()
      self.count = 0
      self.send_motor_commands( self.compute_output() )
      vrep.simxSynchronousTrigger( self.cid )
    else:
      self.update_integral()

  def update_integral( self ):

    for i in range(3):
      self.integrals[i,0] += self.pos_err[i]
      self.integrals[i+3,0] += self.ori_err[i]

  def compute_output( self ):
    """ Computes the rotor velocities based on PID control """

    motor = self.rotor_transform *\
        ( self.gains * self.state + self.I_gain * self.integrals ) +\
        self.gravity_compensation

    return [motor[0,0], motor[1,0], motor[2,0], motor[3,0]]

class PIDt( PID ):
  """ 
  PID controller where the error given to the I term uses velocity as well. 
  It is effectively a task space error.
  """
  def __init__( self, fast_i=False, **kwargs ):

    super(PIDt, self).__init__(**kwargs)

    self.integrals = np.matrix([[0.0],
                                [0.0],
                                [0.0],
                                [0.0],
                               ])

    i1 = 0.001 # Z
    if fast_i: # XY
      i2 = 0.01
    else:
      i2 = 0.0025
    i3 = 0.01 # Yaw

    self.I_gain = np.matrix([[i1, 0, 0, 0],
                             [0, i2, 0, 0],
                             [0, 0, i2, 0],
                             [0, 0, 0, i3],
                            ])
    k6 = 0.802872750102059 * 8
    k8 = 2.8897310746350824 * 4
    
    ak1 = 0.026210965785217845
    ak2 = 2.0 * 13
    ak3 = 0.027614986033826894
    ak4 = 1.65 * 13
    
    self.adaptive_filter = np.matrix([[  0,  0, ak2,   0,   0,-ak4,  0,  0,  0,  0,  0,  0],
                                      [  0, ak1,  0,   0,-ak3,   0,  0,  0,  0,  0,  0,  0],
                                      [-ak1,  0,  0, ak3,   0,   0,  0,  0,  0,  0,  0,  0],
                                      [  0,   0,  0,   0,   0,   0,  0,  0,-k6,  0,  0, k8]
                                     ])

  def update_integral( self ):
    self.integrals += self.adaptive_filter * self.state
    #self.integrals += self.gains * self.state


if __name__ == "__main__":
  cont = PIDt()
  while True:
    cont.control_step()
