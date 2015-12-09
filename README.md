# masters-thesis
Thesis for my masters in Computer Science

Code for the quadcopter controllers used in this thesis are included in the code folder. To run a controller, simply open the quadcopter scene file in V-REP and use the run_model.py script to load the controller. There are 10 controllers available. Their names are: pd, pid, pidt, pidtf, adaptive_transform, angle_correction, basic_transform, non_neural_adaptive, tmc, and tmca.

python run_model.py <controller name>

e.g.

python run_model.py tmca

## Installing Nengo

pip install nengo

## Installing V-REP

Go to http://www.coppeliarobotics.com/downloads.html

Click the appropriate link for your system for V-REP PRO EDU V3.2.1

#### Mac/Linux

Unzip the file and you are good to go. Run it by running the vrep.sh script that it comes with. This can be done by navigating to the unzipped folder and running ./vrep.sh

#### Windows

After you install it there should be an exe you can use to run it

### Setting up the Python API

Instructions can be found [here](http://www.coppeliarobotics.com/helpFiles/en/remoteApiClientSide.htm).
Basically Python needs to know the location of three files (vrep.py, vrepConst.py and remoteApi.extension)

The first two are found in /path/to/vrep/programming/remoteApiBindings/python/python

The last file is in /path/to/vrep/programming/remoteApiBindings/lib/lib/

There is a 32 bit and 64 bit version of this file. I think it has to match with the version of Python you have installed. If one doesn't work, just try the other.

The easiest thing to do is to copy these files into the directory where you are running your Python scripts from. On Mac/Linux you can also just soft link them to your current directory with:

ln -s /path/to/file/filename.extension

A list of available commands you can use from Python can be found [here](http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsPython.htm) and organized by category [here](http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionListCategory.htm)

### Scene Files for the Quadcopter

This is either the quadcopter_experiments_simple.ttt or quadcopter_demo.ttt file, attached to the release (there's a button near the top of the github page to access it). The former is a blank environment used for testing, the latter is a more interesting environment that can also display some plots (but will run slower). To use the plots, run the demo.py file.
