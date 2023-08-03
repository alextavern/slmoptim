from slmOptim.optim import transmission_matrix

# some parameters
order = 4
mag = 4

middle_x = int(1440 / 2)
middle_y = int(1080 / 2)
dim = 100
roi = (middle_x - dim, middle_y - dim, middle_x + dim, middle_y + dim)

bins = 4

exposure_time = 400
gain = 1
timeout= 100 # timeout

# measure tm
tm_raw = transmission_matrix.measTM(roi=roi, 
                            bins=bins, 
                            exposure_time=exposure_time, 
                            order=order, 
                            mag=mag)

patterns, frames = tm_raw.get_tm()
tm_raw.save_tm()

# calculate tm
tr = transmission_matrix.calcTM(frames)
_, _, _, tm = tr.calc_plot_tm()

# inverse