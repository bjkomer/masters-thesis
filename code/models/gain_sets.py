# Easy access for gain sets. Reusable code
import numpy as np

def get_gain_matrices( gain_set='hybrid' ):
  if gain_set == 'old':
    # Original Gains
    k1=0.22#0.21           # XY
    k2=2.0            # Z
    k3=np.sqrt(k1)  # dXY
    k4=1.65           # dZ
    k5=3.80#3.75           # RP
    k6=10             # Yaw
    k7=np.sqrt(k5)  # dRP
    k8=np.sqrt(k6)  # dYaw
    
    ak1=0.21*.03 # XY
    ak2=2.0*5   # Z
    ak3=0.5*.03  # dXY
    ak4=1.65*5  # dZ

  elif gain_set == 'hyperopt':
    # The gains hyperopt picked
    k1 = 0.43352026190263104
    k2 = 3.8617161507986935
    k3 = 0.5388202808181405
    k4 = 4.67017496312439
    k5 = 2.5995452450850185
    k6 = 0.802872750102059
    k7 = 0.5990281657438163
    k8 = 2.8897310746350824
    
    ak1 = 0.026210965785217845
    ak2 = 48.2387479238364
    ak3 = 0.027614986033826894
    ak4 = 34.96264477140544

  elif gain_set == 'hybrid':
    # A hybrid between hyperopt and manual tuning
    k1 = 0.43352026190263104
    k2 = 2.0 * 2
    k3 = 0.5388202808181405
    k4 = 1.65 * 2
    k5 = 2.5995452450850185
    k6 = 0.802872750102059 * 8
    k7 = 0.5990281657438163
    k8 = 2.8897310746350824 * 4
    
    ak1 = 0.026210965785217845
    ak2 = 2.0 * 5
    ak3 = 0.027614986033826894
    ak4 = 1.65 * 5
  elif gain_set == 'hybrid_fast':
    # A hybrid between hyperopt and manual tuning, faster z adaptation
    k1 = 0.43352026190263104
    k2 = 2.0 * 4
    k3 = 0.5388202808181405
    k4 = 1.65 * 4
    k5 = 2.5995452450850185
    k6 = 0.802872750102059 * 8
    k7 = 0.5990281657438163
    k8 = 2.8897310746350824 * 4
    
    ak1 = 0.026210965785217845
    ak2 = 2.0 * 13
    ak3 = 0.027614986033826894
    ak4 = 1.65 * 13
  elif gain_set == 'wind':
    # Focus on adapting fast to wind
    k1 = 0.43352026190263104
    k2 = 2.0 * 4
    k3 = 0.5388202808181405 * 1.25
    k4 = 1.65 * 4
    k5 = 2.5995452450850185
    k6 = 0.802872750102059 * 8
    k7 = 0.5990281657438163
    k8 = 2.8897310746350824 * 4
    
    ak1 = 0.026210965785217845 * 6
    ak2 = 2.0 * 13
    ak3 = 0.027614986033826894 * 5.5
    ak4 = 1.65 * 13

  gain_matrix = np.matrix([[ 0,  0, k2,  0,  0,-k4,  0,  0,  0,  0,  0,  0],
                          [  0, k1,  0,  0,-k3,  0,-k5,  0,  0, k7,  0,  0],
                          [-k1,  0,  0, k3,  0,  0,  0,-k5,  0,  0, k7,  0],
                          [  0,  0,  0,  0,  0,  0,  0,  0,-k6,  0,  0, k8] 
                          ])

  adaptive_filter = np.matrix([[ 0,  0, ak2,  0,  0,-ak4,  0,  0,  0,  0,  0,  0],
                              [  0, ak1,  0,  0,-ak3,  0,  0,  0,  0,  0,  0,  0],
                              [-ak1,  0,  0, ak3,  0,  0,  0,  0,  0,  0,  0,  0],
                              [  0,   0,  0,  0,   0,  0,  0,  0,-k6,  0,  0, k8]
                              ])
  
  # Defined to match the alignment of the propellors
  task_to_rotor = np.matrix([[ 1,-1, 1, 1],
                             [ 1,-1,-1,-1],
                             [ 1, 1,-1, 1],
                             [ 1, 1, 1,-1] ])

  return gain_matrix, adaptive_filter, task_to_rotor
