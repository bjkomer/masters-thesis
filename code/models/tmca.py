# Neural Adaptive TMCA

# Uses Target-Modulated-Control on the regular and angle adaptation ensembles
# Allocentric information included in state
import nengo
import gain_sets
import numpy as np
from quadcopter import FullStateTargetQuadcopter

class Model( object ):

  def __init__( self, target_func, cid=None, decoder_solver=None,
                noise_std=None):

    gain_matrix, adaptive_filter, task_to_rotor = gain_sets.get_gain_matrices( gain_set='hybrid_fast' )
    
    N_ADAPT = 3000
    N_ANGLE_ADAPT = 3000

    learning_rate_standard = 1e-7 /3
    learning_rate_angle = 1e-7 /3

    k1 = 0.43352026190263104
    k2 = 2.0 * 2
    k3 = 0.5388202808181405
    k4 = 1.65 * 2
    k5 = 2.5995452450850185
    k6 = 0.802872750102059 * 2
    k7 = 0.5990281657438163
    k8 = 2.8897310746350824 * 2

    k1 = k1 * .1
    k3 = k3 * .11

    def target_modulation_angle( x ):
      # Simple version
      roll = k1 * x[1] - k3 * x[4]
      pitch = k1 * x[0] - k3 * x[3]
      # Modulation from target
      th = .1
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

    self.model = nengo.Network( label='V-REP Adaptive Quadcopter', seed=13 )
    with self.model:
      
      # Full state adaptation
      # Sensors and Actuators
      self.copter_node = FullStateTargetQuadcopter( target_func=target_func,
                                                    cid=cid, noise_std=noise_std )
      copter = nengo.Node(self.copter_node, size_in=4, size_out=24)

      # State Error Population
      state = nengo.Ensemble(n_neurons=1, dimensions=18, neuron_type=nengo.Direct())
      
      target = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
      
      # A slower moving representation of the target
      delayed_target = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
      nengo.Connection(target, delayed_target, synapse=1.0)

      # Difference between target and delayed target
      # When this is non-zero, it means the target has moved recently
      target_difference = nengo.Ensemble(n_neurons=1, dimensions=6, neuron_type=nengo.Direct())
      
      nengo.Connection(target, target_difference, synapse=None)
      # TODO: could possibly remove the delayed target population and just have 2
      # connections coming from target with different synapses
      nengo.Connection(delayed_target, target_difference, synapse=None, transform=-1)

      # Contains the rotor speeds
      motor = nengo.Ensemble(n_neurons=1, dimensions=4, neuron_type=nengo.Direct())
      
      # Command in 'task' space (up/down, forward/back, left/right, rotate)
      task = nengo.Ensemble(n_neurons=1, dimensions=4, neuron_type=nengo.Direct())
      
      adaptation = nengo.Ensemble(n_neurons=N_ADAPT, dimensions=18)

      nengo.Connection(state, adaptation, synapse=None)


      # Angle Correction
      angle_adapt = nengo.Ensemble(n_neurons=N_ANGLE_ADAPT, dimensions=18)
      corrected_state = nengo.Ensemble(n_neurons=1, dimensions=12, neuron_type=nengo.Direct())
      angle_correction = nengo.Ensemble(n_neurons=1, dimensions=2,
                                        neuron_type=nengo.Direct())
      
      # Combined state population with the target change population for modulation
      state_with_target = nengo.Ensemble(n_neurons=1, dimensions=18, neuron_type=nengo.Direct())
      nengo.Connection(state, angle_adapt, synapse=None)
      nengo.Connection(state[:12], corrected_state, synapse=None)
      nengo.Connection(angle_correction, corrected_state[[6,7]], synapse=None)
      #nengo.Connection(angle_correction, corrected_state[[6,7]], synapse=0.1)
      nengo.Connection(state[:12], state_with_target[:12], synapse=None)
      nengo.Connection(target_difference, state_with_target[12:], synapse=None)
      
      # Standard Adaptive Population
      error_conn = nengo.Connection(state_with_target, task, 
                                    function=target_modulation_standard,
                                    modulatory=True)
      
      if decoder_solver is None:
        self.a_conn = nengo.Connection(adaptation, task, function=lambda x: [0,0,0,0],
                         learning_rule_type=nengo.PES(error_conn,
                                                      learning_rate=learning_rate_standard))
      else:
        self.a_conn = nengo.Connection(adaptation, task, 
                                       function=lambda x: [0,0,0,0],
                                       solver=decoder_solver[0],
                         learning_rule_type=nengo.PES(error_conn,
                                                      learning_rate=learning_rate_standard))
      
      # Angle Correction Adaptive Population
      error_conn = nengo.Connection(state_with_target, angle_correction,
                                    function=target_modulation_angle,
                                    modulatory=True)
      
      if decoder_solver is None:
        self.aa_conn = nengo.Connection(angle_adapt, angle_correction, function=lambda x: [0,0],
                         learning_rule_type=nengo.PES(error_conn,
                                                      learning_rate=learning_rate_angle))
      else:
        self.aa_conn = nengo.Connection(angle_adapt, angle_correction,
                                        function=lambda x: [0,0],
                                        solver=decoder_solver[1],
                         learning_rule_type=nengo.PES(error_conn,
                                                      learning_rate=learning_rate_angle))

      nengo.Connection(corrected_state, task, transform=gain_matrix)
      nengo.Connection(task, motor, transform=task_to_rotor)
      nengo.Connection(copter[:18], state, synapse=None)
      nengo.Connection(copter[18:], target, synapse=None)
      nengo.Connection(motor, copter, synapse=0.001)

  def get_model( self ):
    return self.model

  def get_copter( self ):
    return self.copter_node

  def get_learned_connections( self ):
    return [self.a_conn, self.aa_conn]
