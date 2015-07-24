# Neural Adaptive Angle Correction

# Uses adaptation for angle correction as well as regular control
import nengo
import gain_sets
import numpy as np
from quadcopter import Quadcopter


class Model( object ):

  def __init__( self, target_func, cid=None, decoder_solver=None,
                noise_std=None):

    gain_matrix, adaptive_filter, task_to_rotor = gain_sets.get_gain_matrices( gain_set='hybrid_fast' )

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
    angle_adapt_filter = np.matrix([[  0, -k1,  0,  0, k3,  0,  0,  0,  0,  0,  0,  0],
                                    [ k1,  0,  0, -k3,  0,  0,  0,  0,  0,  0,  0,  0],
                                   ])
    self.model = nengo.Network( label='V-REP Adaptive Quadcopter', seed=13 )
    with self.model:
      
      # Sensors and Actuators
      self.copter_node = Quadcopter(target_func=target_func, cid=cid,
                                    noise_std=noise_std)
      copter = nengo.Node(self.copter_node, size_in=4, size_out=12)

      # State Error Population
      state = nengo.Ensemble(n_neurons=1, dimensions=12, neuron_type=nengo.Direct())
      
      # Contains the rotor speeds
      motor = nengo.Ensemble(n_neurons=1, dimensions=4, neuron_type=nengo.Direct())
      
      # Command in 'task' space (up/down, forward/back, left/right, rotate)
      task = nengo.Ensemble(n_neurons=1, dimensions=4, neuron_type=nengo.Direct())
      
      adaptation = nengo.Ensemble(n_neurons=1000, dimensions=12)

      nengo.Connection(state, adaptation, synapse=None)

      error_conn = nengo.Connection(state, task, transform=adaptive_filter,
      #error_conn = nengo.Connection(state, task, transform=gain_matrix,
                                    modulatory=True)
      
      if decoder_solver is None:
        self.a_conn = nengo.Connection(adaptation, task, function=lambda x: [0,0,0,0],
                         learning_rule_type=nengo.PES(error_conn,
                                                      learning_rate=1e-7))
      else:
        self.a_conn = nengo.Connection(adaptation, task, solver=decoder_solver[0],
                         learning_rule_type=nengo.PES(error_conn,
                                                      learning_rate=1e-7))

      # Angle Correction
      angle_adapt = nengo.Ensemble(n_neurons=1000, dimensions=12)
      corrected_state = nengo.Ensemble(n_neurons=1, dimensions=12, neuron_type=nengo.Direct())
      angle_correction = nengo.Ensemble(n_neurons=1, dimensions=2,
                                        neuron_type=nengo.Direct())
      nengo.Connection(state, angle_adapt, synapse=None)
      nengo.Connection(state, corrected_state, synapse=None)
      nengo.Connection(angle_correction, corrected_state[[6,7]], synapse=None)
      
      error_conn = nengo.Connection(state, angle_correction, transform=angle_adapt_filter,
                                    modulatory=True)
      
      if decoder_solver is None:
        self.aa_conn = nengo.Connection(angle_adapt, angle_correction, function=lambda x: [0,0],
                         learning_rule_type=nengo.PES(error_conn,
                                                      learning_rate=1e-7))
      else:
        self.aa_conn = nengo.Connection(angle_adapt, angle_correction, solver=decoder_solver[1],
                         learning_rule_type=nengo.PES(error_conn,
                                                      learning_rate=1e-7))

      nengo.Connection(corrected_state, task, transform=gain_matrix)
      nengo.Connection(task, motor, transform=task_to_rotor)
      nengo.Connection(copter, state, synapse=None)
      nengo.Connection(motor, copter, synapse=0.001)

  def get_model( self ):
    return self.model

  def get_copter( self ):
    return self.copter_node

  def get_learned_connections( self ):
    return [self.a_conn, self.aa_conn]
