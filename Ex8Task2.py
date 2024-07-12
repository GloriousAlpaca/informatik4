import numpy as np

# Define Environment
enviro = np.full((3,3), -1)
# Define Endgoal
enviro[2,2] = 10
# Define r
enviro[2,0] = 3
gamma = 0.5

utilities = enviro.copy()

prob = [0,0,0]