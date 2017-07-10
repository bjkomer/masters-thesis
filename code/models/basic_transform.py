# Simple Neural Adaptive

# Uses adaptation for regular control
import nengo
import gain_sets
from quadcopter import Quadcopter

class Model( object ):

  def __init__( self, target_func, cid=None, decoder_solver=None,
                noise_std=None ):

    gain_matrix, adaptive_filter, task_to_rotor = gain_sets.get_gain_matrices( gain_set='hybrid_fast' )
    
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

      if decoder_solver is None:
        self.a_conn = nengo.Connection(adaptation, task, function=lambda x: [0,0,0,0],
                         learning_rule_type=nengo.PES(learning_rate=1e-4))
      else:
        self.a_conn = nengo.Connection(adaptation, task, function=lambda x: [0,0,0,0],
                                       solver=decoder_solver[0],
                         learning_rule_type=nengo.PES(learning_rate=1e-4))
      
      # Sign of the error changed in newer versions of Nengo since this work
      error_conn = nengo.Connection(state, self.a_conn.learning_rule,
                                    transform=-1*gain_matrix)

      nengo.Connection(state, task, transform=gain_matrix)
      nengo.Connection(task, motor, transform=task_to_rotor)
      nengo.Connection(copter, state, synapse=None)
      nengo.Connection(motor, copter, synapse=0.001)

  def get_model( self ):
    return self.model

  def get_copter( self ):
    return self.copter_node
  
  def get_learned_connections( self ):
    return [self.a_conn]
