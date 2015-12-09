# Demo for adaptive quadcopter using target modulated control
# TODO: code is super gross right now, clean it up some time later
import nengo
import vrep
import numpy as np
import subprocess
import sys
import signal
import os
import time

from quadcopter import FullStateTargetQuadcopter
from vrep_utils import SendData

# TODO: put vrep nodes into nengo.utils.vrep and use them from there

VREP_PLOTS = True

# If noise is being modelled. Build the circuit slightly differently to account
# for it (i.e. add filters)
NOISE = False

gain_set = 'hybrid_fast'
if len(sys.argv) == 2:
  if sys.argv[1] in ['old', 'hyperopt', 'hybrid', 'hybrid_fast', 'wind']:
    gain_set = sys.argv[1]
    LIVE_PLOTS = False
  else:
    LIVE_PLOTS = True
elif len(sys.argv) == 3:
  if sys.argv[1] in ['old', 'hyperopt', 'hybrid', 'hybrid_fast', 'wind']:
    gain_set = sys.argv[1]
    LIVE_PLOTS = True
  else:
    LIVE_PLOTS = True
else:
  LIVE_PLOTS = False

real_time = True
dt = 0.001 


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
  #k2 = -3.8617161507986935 # hyperopt picked a negative gain for some reason
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
  # Focus on adaptive fast to wind
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
                        [  0,  0,  0,  0,  0,  0,  0,  0,-k6,  0,  0, k8] ])

adaptive_filter = np.matrix([[ 0,  0, ak2,  0,  0,-ak4,  0,  0,  0,  0,  0,  0],
                            [  0, ak1,  0,  0,-ak3,  0,  0,  0,  0,  0,  0,  0],
                            [-ak1,  0,  0, ak3,  0,  0,  0,  0,  0,  0,  0,  0],
                            [  0,  0,  0,  0,  0,  0,  0,  0,-k6,  0,  0, k8]
                            ])

# Defined to match the alignment of the propellors
task_to_rotor = np.matrix([[ 1,-1, 1, 1],
                           [ 1,-1,-1,-1],
                           [ 1, 1,-1, 1],
                           [ 1, 1, 1,-1] ])

k1 = 0.43352026190263104
k2 = 2.0 * 2
k3 = 0.5388202808181405
k4 = 1.65 * 2
k5 = 2.5995452450850185
k6 = 0.802872750102059 * 2
k7 = 0.5990281657438163
k8 = 2.8897310746350824 * 2

k1 = k1 * .15
k3 = k3 * .15

def target_modulation( x ):
  # Simple version
  roll = k1 * x[1] - k3 * x[4]
  pitch = k1 * x[0] - k3 * x[3]
  # Modulation from target
  th = .1
  #print("x: %.2f, y: %.2f" % (x[12],x[13]))
  if abs(x[13]) > th:
    roll = 0
  if abs(x[12]) > th:
    pitch = 0
  return [-roll, pitch]

def target_modulation_standard( x ):
  state = np.matrix([[x[0]],
                     [x[1]],
                     [x[2]],
                     [x[3]],
                     [x[4]],
                     [x[5]],
                     [x[6]],
                     [x[7]],
                     [x[8]],
                     [x[9]],
                     [x[10]],
                     [x[11]],
                    ])
  task = adaptive_filter * state
  # Modulation from target
  th = .1
  if abs(x[14]) > th:
    task[0,0] = 0
  if abs(x[13]) > th:
    task[1,0] = 0
  if abs(x[12]) > th:
    task[2,0] = 0
  if abs(x[17]) > th:
    task[3,0] = 0
  return [task[0,0], task[1,0], task[2,0], task[3,0]]

model = nengo.Network( label='V-REP Adaptive Quadcopter TMC', seed=13 )
with model:
  
  # Full state adaptation
  # Sensors and Actuators
  if NOISE:
    copter_node = FullStateTargetQuadcopter(noise=True, noise_std=[.05,.05,.05,.05])
  else:
    copter_node = FullStateTargetQuadcopter()
  copter = nengo.Node(copter_node, size_in=4, size_out=24)

  # State Error Population
  state = nengo.Ensemble(n_neurons=1, dimensions=12, neuron_type=nengo.Direct())
  allo_state = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
  
  target = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
  
  # A slower moving representation of the target
  delayed_target = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
  nengo.Connection(target, delayed_target, synapse=1.0)
  
  minor_delayed_target = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
  nengo.Connection(target, minor_delayed_target, synapse=0.01)

  # Difference between target and delayed target
  # When this is non-zero, it means the target has moved recently
  target_difference = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
  minor_target_difference = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
  
  nengo.Connection(target, target_difference, synapse=None)
  nengo.Connection(target, minor_target_difference, synapse=None)
  # TODO: remove the delayed target population and just have 2
  # connections coming from target with different synapses
  nengo.Connection(delayed_target, target_difference, synapse=None, transform=-1)
  nengo.Connection(minor_delayed_target, minor_target_difference, synapse=None, transform=-1)

  # Contains the rotor speeds
  motor = nengo.Ensemble(n_neurons=1, dimensions=4, neuron_type=nengo.Direct())
  
  # Command in 'task' space (up/down, forward/back, left/right, rotate)
  task = nengo.Ensemble(n_neurons=1, dimensions=4, neuron_type=nengo.Direct())
  
  adaptation = nengo.Ensemble(n_neurons=1000, dimensions=12)

  nengo.Connection(state, adaptation, synapse=None)

  # Angle Correction
  angle_adapt = nengo.Ensemble(n_neurons=1000, dimensions=12)
  corrected_state = nengo.Ensemble(n_neurons=1, dimensions=12, neuron_type=nengo.Direct())
  angle_correction = nengo.Ensemble(n_neurons=1, dimensions=2,
                                    neuron_type=nengo.Direct())
  
  # Combined state population with the target change population for modulation
  state_with_target = nengo.Ensemble(n_neurons=1, dimensions=18, neuron_type=nengo.Direct())
  minor_state_with_target = nengo.Ensemble(n_neurons=1, dimensions=18, neuron_type=nengo.Direct())
  nengo.Connection(state, angle_adapt, synapse=None)
  nengo.Connection(state, corrected_state, synapse=None)
  nengo.Connection(angle_correction, corrected_state[[6,7]], synapse=None)
  nengo.Connection(state, state_with_target[:12], synapse=None)
  nengo.Connection(state, minor_state_with_target[:12], synapse=None)
  nengo.Connection(target_difference, state_with_target[12:], synapse=None)
  nengo.Connection(minor_target_difference, minor_state_with_target[12:], synapse=None)
  

  # Standard Adaptive Population
  error_conn = nengo.Connection(minor_state_with_target, task, 
                                function=target_modulation_standard,
                                modulatory=True)
  
  nengo.Connection(adaptation, task, function=lambda x: [0,0,0,0],
                   learning_rule_type=nengo.PES(error_conn,
                                                learning_rate=1e-7),) 


  # Angle Correction Adaptive Population
  error_conn_angle = nengo.Connection(state_with_target, angle_correction,
                                function=target_modulation,
                                modulatory=True)
  nengo.Connection(angle_adapt, angle_correction, function=lambda x: [0,0],
                   learning_rule_type=nengo.PES(error_conn_angle,
                                                learning_rate=1e-7))

  nengo.Connection(corrected_state, task, transform=gain_matrix)
  nengo.Connection(task, motor, transform=task_to_rotor)
  nengo.Connection(copter[:12], state, synapse=None)
  nengo.Connection(copter[12:18], allo_state, synapse=None)
  nengo.Connection(copter[18:], target, synapse=None)
  nengo.Connection(motor, copter, synapse=0.001)
 
  # Send data to the plots in V-REP
  if VREP_PLOTS:
    state_display = nengo.Node( SendData(signal_name="state"),
                                size_in=6, size_out=0 )
    motor_display = nengo.Node( SendData(signal_name="rotor"),
                                size_in=4, size_out=0 )
    spikes_display = nengo.Node( SendData(signal_name="spikes", size=100),
                                size_in=1000, size_out=0 )
    nengo.Connection( adaptation.neurons, spikes_display, synapse=None )
    nengo.Connection( motor, motor_display, synapse=None )
    nengo.Connection( state[[0,1,2,6,7,8]], state_display, synapse=None )


print( "starting simulator..." )
before = time.time()

sim = nengo.Simulator( model, dt=dt )

after = time.time()
print( "time to build:" )
print( after - before )

print( "running simulator..." )
before = time.time()

# Start the GUI
#subprocess.Popen(["python", "./vrep_gui.py", str(os.getpid())])

state = []

def reset(signal, frame):
  # Save State Information
  from run_model import save
  save("benchmarks/demo.p", state)
  copter_node.reset()
  sim.reset()
signal.signal(signal.SIGUSR1, reset)
for i in range( int(10000 / dt) ):
  sim.step()
  # Record State
  state.append(copter_node.get_state())

after = time.time()
print( "time to run:" )
print( after - before )
