import matplotlib.pyplot as plt
import numpy as np
from slmOptim.optim import transmission_matrix
from slmOptim.zeluxPy import helper_functions as cam
from slmOptim.patternSLM import patterns as pt
from slmPy import slmpy


# some parameters
order = 4
mag = 4

middle_x = int(1440 / 2)
middle_y = int(1080 / 2)
dim = 100
roi = (middle_x - dim, middle_y - dim, middle_x + dim, middle_y + dim)

speckele_grain = 4
bins = speckele_grain

exposure_time = 400
gain = 1
timeout= 100 # timeout


# tm = transmission_matrix.TM(roi=roi, 
#                             bins=bins, 
#                             exposure_time=exposure_time, 
#                             order=order, 
#                             mag=mag)

# tm_raw = tm.get_tm()