#--------------------- importing the required libraries -------------
from IPython.display import clear_output
import numpy as np

from tensorflow.compat.v1.keras.optimizers import Adam
import matplotlib.pyplot as plt

from options.EuropeanPut import EuropeanPut
from vanillaAlgo.vanilla_generalAlgo import vanillaModel
from functions.losses import Entropy
from functions.splitVanilla import set_split_vanilla

# need to import multiGeom dynamics
# need to import Kou Jump Diffusion dynamics
#---------------------------------------------------------------------
clear_output()
print("\nFinish installing and importing all necessary libraries!")
#------------------- initializing our constants ----------------------

#---------------------------------------------------------------------
# need to call our process
# need to store our paths

clear_output()
#---------------------------------------------------------------------
finalPayoff
tradingSet
infoSet

# need to divide the data into training and testing set
#----------------------------------------------------------------------
print("Finish preparing data!")
#--------------------- running the algorithm --------------------------
# call, compile and fit the model
#----------------------------------------------------------------------
clear_output()

print("Finished running deep hedging algorithm! (Simple Network)")
#---------------------- benchmark comparison --------------------------

#----------------------------------------------------------------------
#------------------- building the comparison graph --------------------

#------------------------------------------------------------------------