import util as util
import matplotlib.pyplot as plt


#Hyperbola Example 4.5 from Curtis Orbital Mechanics textbook 
duration = 3600.0*(0.1)
initial_state = {"a (m)": -16725186.345,
                "e": 1.4 ,
                 "i (deg)": 30.0 ,
                 "RAAN (deg)": 40.0 ,
                 "aop (deg)": 60.0, 
                 "M (deg)" : 5.176237274033764}
result = util.OrbitalSimulation(duration, initial_state, [2024, 5, 22, 20, 35, 59.34825], 360.0, False, "Earth")
print(result.solution_ECI[0,:])

#############################################################################

#Plot
plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(result.solution_ECI[:,0], result.solution_ECI[:,1], result.solution_ECI[:,2], 'blue')
plt.show()
