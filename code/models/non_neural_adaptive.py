# Non-Neural Adaptive

import nengo
import gain_sets
from quadcopter import Quadcopter, AdaptiveController, EnhancedAdaptiveController

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
    
      controller = nengo.Node(AdaptiveController(), size_in=12, size_out=4)

      nengo.Connection(copter, controller, synapse=None)
      nengo.Connection(controller, copter, synapse=0.001)

  def get_model( self ):
    return self.model

  def get_copter( self ):
    return self.copter_node

  def get_learned_connections( self ):
    raise NotImplemented, "There are no decoders as this is a non-neural controller"
