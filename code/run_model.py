# Runs a selected quadcopter controller model
# This controller must exist in the 'models' directory

import sys
import time
import nengo
import subprocess
import os
import signal
from controller import PD, PID, PIDt
import cPickle as pickle
import numpy as np

dt = 0.001

# Load a model based on the string name
def load_model( name, target_func=None, cid=None, decoders=None,
                noise_std=None):
  exec( "import models.%s as Model" % name )
  return Model.Model(target_func, cid, decoders, noise_std=noise_std)

# Define target movement functions for benchmarking
# 3 seconds are given to allow the quadcopter to stabilize at the start point
def simple_vertical( step, dist=2.0 ):
  if step < 300:
    return [0,0,0.5], [0,0,0]
  else:
    return [0,0,0.5 + dist], [0,0,0]

def simple_horizontal( step, dist=1.0 ):
  if step < 300:
    return [0,0,0.5], [0,0,0]
  else:
    return [dist,0,0.5], [0,0,0]

def simple_rotational( step, dist=3.14159/2.0 ):
  if step < 300:
    return [0,0,0.5], [0,0,0]
  else:
    return [0,0,0.5], [0,0,dist]

def simple_wind( step, dist=1.0 ):
  if step < 300:
    return [0,0,0.5], [0,0,0]
  else:
    return [0,dist,0.5], [0,0,0]

# Continuously moves back and forth between two points forever
def loop_horizontal( step, dist=2.0 ):
  if int(step / 500.0) % 2 == 0:
    return [0,0,0.5], [0,0,0]
  else:
    return [dist,0,0.5], [0,0,0]

def load_target_func( name ):
  if name == 'vertical':
    return simple_vertical
  elif name == 'horizontal':
    return simple_horizontal
  elif name == 'rotational':
    return simple_rotational
  elif name == 'wind':
    return simple_wind
  elif name == 'loop_horizontal':
    return loop_horizontal
  else:
    return simple_vertical

def parse_state(state):
  l = []
  for group in state:
    for dim in group:
      l.append(dim)
  return np.array(l)

def save(fname, state):
    data = np.zeros((len(state), 18))

    for i,s in enumerate(state):
      data[i] = parse_state(s)

    pickle.dump(data, open(fname, 'wb'))

if __name__ == "__main__":

  num_steps = 17000
  target_func = None
  model_name = 'default'
  target_name = 'custom'
  recording = True

  if len(sys.argv) == 2:
    model_name = sys.argv[1]
    recording = False # don't record to file if no target function is specified
  elif len(sys.argv) == 3:
    model_name = sys.argv[1]
    target_name = sys.argv[2]
    target_func = load_target_func(target_name)
  elif len(sys.argv) == 4:
    model_name = sys.argv[1]
    target_name = sys.argv[2]
    target_func = load_target_func(target_name)
    num_steps = int(sys.argv[3])

  if model_name in ['pd', 'pid', 'pidt', 'pidtf']:
    # The controller does not use a Nengo Model
    
    if model_name == 'pd':
      cont = PD( target_func = target_func )
    elif model_name == 'pid':
      cont = PID( target_func = target_func )
    elif model_name == 'pidt':
      cont = PIDt( target_func = target_func )
    elif model_name == 'pidtf':
      cont = PIDt( fast_i=True, target_func = target_func )
    else:
      cont = PIDt( target_func = target_func )

    state = []
    count = 0

    # if no target is given, run 'forever'
    # also don't bother saving anything
    if not recording:
      while True:
        cont.control_step()
    else:
      for i in range( num_steps ):
        cont.control_step()
        count += 1
        # Match the recording to the nengo models
        if count == 10:
          count = 0
          state.append(cont.get_state())
      
      fname = "benchmarks/%s_%s_%s.p" % (model_name, target_name, num_steps)

      save(fname, state)


  else:
    # Build and run a Nengo Model

    exec( "import models.%s as Model" % model_name )
    m = Model.Model( target_func )
    #m = Model.Model( target_func,noise_std=[.2,.2,.02,.02] )
    model = m.get_model()
    #model = Model.get_model( target_func )

    fname = 'benchmarks/%s_%s_%s.p' % (model_name, target_name, num_steps)

    print( "starting simulator..." )
    before = time.time()

    sim = nengo.Simulator( model, dt=dt )

    after = time.time()
    print( "time to build:" )
    print( after - before )

    print( "running simulator..." )
    before = time.time()
    
    def reset(signal, frame):
      # Save State Information
      if recording:
        benchmark.save(fname, state)
      m.get_copter().reset()
      sim.reset()

    signal.signal(signal.SIGUSR1, reset)

    state = []
    count = 0

    # Run until cancelled if not recording
    if recording:
      for i in range( num_steps ):
        sim.step()
        count += 1
        # Record state every sim_dt/dt steps
        if count == 10:
          count = 0
          state.append(m.get_copter().get_state())
      save(fname, state)
    else:
      while True:
        sim.step()

    after = time.time()
    print( "time to run:" )
    print( after - before )
    print("Results saved to: %s" % fname)
