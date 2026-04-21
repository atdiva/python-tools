import util as util
import numpy as np
import matplotlib.pyplot as plt

#Generalized Lotka-Volterra equations
def f_derivative_LV_generalized(t, xy, genparams):

    #parameters to integrate
    x = xy[0] #prey
    y = xy[1] #predator
    #additional parameters
    a = genparams[0]
    b = genparams[1]
    c = genparams[2]
    d = genparams[3]
    e = genparams[4]
    f = genparams[5]
    g = genparams[6]
    h = genparams[7]
    i = genparams[8]
    j = genparams[9]

    return np.array( [ x*a + b*y*x + e*y + g*(x**2) + i , y*c + d*x*y + f*x + h*(y**2) + j] )

###########################################################################################
#time vector
time = np.linspace(0, 150, 10**4)

#Example 1: Sim1 https://teaching.smp.uq.edu.au/scims/Appl_analysis/Lotka_Volterra.html
#params = [0.1, -0.002, -0.2, 0.0025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#init_xy = [80, 20]

#Example 2: Sim2 https://teaching.smp.uq.edu.au/scims/Appl_analysis/Lotka_Volterra.html
params = [1.0, -0.02, -0.25, 0.02, 0.0, 0.0, -0.01, 0.0, 0.0, -5.0]
init_xy = [20, 20]

#Run Integration 
solution = util.rk4(time, f_derivative_LV_generalized, init_xy, params)

###########################################################################################
plt.close('all')
plt.figure(dpi=100)
plt.subplot(121)
plt.title("Population vs Time")
plt.plot(time, solution[:,0], label="prey")
plt.plot(time, solution[:,1], label="predator")
plt.grid()
plt.legend()
plt.xlabel("Time")
plt.ylabel("Population Size")
plt.subplot(122)
plt.title("Phase Plot (x-prey,y-predator)")
plt.plot(solution[:,0], solution[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.tight_layout()
plt.show()

