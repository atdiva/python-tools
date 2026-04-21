import util as util
import matplotlib.pyplot as plt
import pandas as pd

#Data drawn from 
#3I/ATLAS: https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=1004083&view=OPC
#Planets: https://ssd.jpl.nasa.gov/horizons/app.html#/
def set_initial_state(dataframe, object_index_row):
    #a, e, i, o, w, M
    output = {dataframe.columns[3] : dataframe[dataframe.columns[3]][object_index_row] ,
              dataframe.columns[4] : dataframe[dataframe.columns[4]][object_index_row] ,	
              dataframe.columns[5] : dataframe[dataframe.columns[5]][object_index_row] ,
              dataframe.columns[6] : dataframe[dataframe.columns[6]][object_index_row] ,
              dataframe.columns[7] : dataframe[dataframe.columns[7]][object_index_row] ,
              dataframe.columns[8] : dataframe[dataframe.columns[8]][object_index_row] }
    return output

#############################################################################

#constants and duration 
one_AU_in_m = 149597870691 #m
one_day = 3600*24 #sec 
duration = 200*one_day #from time stamp of KO state

#read data, note error due to osculating elements with no model 
df = pd.read_excel("data/Astro/Ephem/OrbitalElements.xlsx", sheet_name=0)
indx_3I = df[ df['Name'] == "3I/ATLAS"].index[0]
indx_Earth = df[df['Name'] == "Earth"].index[0]
indx_Mars = df[df['Name'] == "Mars"].index[0]
indx_Jupiter = df[df['Name'] == "Jupiter"].index[0]

#SBDL JPL: https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=1004083
initial_state_3I = set_initial_state(df, indx_3I)

#https://ssd.jpl.nasa.gov/horizons/app.html#/
initial_state_Earth = set_initial_state(df, indx_Earth)
initial_state_Mars = set_initial_state(df, indx_Mars)
initial_state_Jupiter = set_initial_state(df, indx_Jupiter)

#############################################################################

result_3I = util.OrbitalSimulation(duration, initial_state_3I, "", one_day/100, False, "Sun")
result_Earth = util.OrbitalSimulation(duration, initial_state_Earth, "", one_day/100, False, "Sun")
result_Mars = util.OrbitalSimulation(duration, initial_state_Mars, "", one_day/100, False, "Sun")
result_Jupiter = util.OrbitalSimulation(duration, initial_state_Jupiter, "", one_day/100, False, "Sun")

print("---------------------------------------------")
print("Time to Perihelion (days from epoch) : " + str( (1/one_day)*result_3I.t[ util.find_index_of_nearest(result_3I.magr, min(result_3I.magr)) ] ) )
print("Perihelion Distance (AU) : " + str( (1/one_AU_in_m)*result_3I.magr[ util.find_index_of_nearest(result_3I.magr, min(result_3I.magr)) ] ) )
print("Max speed v (km/s) : " + str( (1/1000)*result_3I.magv[ util.find_index_of_nearest(result_3I.magv, max(result_3I.magv)) ] ) )

#Plot
plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(result_3I.solution_CCI[:,0], result_3I.solution_CCI[:,1], result_3I.solution_CCI[:,2], 'blue')
ax.plot3D(result_Earth.solution_CCI[:,0], result_Earth.solution_CCI[:,1], result_Earth.solution_CCI[:,2], 'green')
ax.plot3D(result_Mars.solution_CCI[:,0], result_Mars.solution_CCI[:,1], result_Mars.solution_CCI[:,2], 'k')
ax.plot3D(result_Jupiter.solution_CCI[:,0], result_Jupiter.solution_CCI[:,1], result_Jupiter.solution_CCI[:,2], 'orange')
ax.scatter(result_3I.solution_CCI[0,0], result_3I.solution_CCI[0,1], result_3I.solution_CCI[0,2], 'b', s = 75)
ax.scatter(result_Earth.solution_CCI[0,0], result_Earth.solution_CCI[0,1], result_Earth.solution_CCI[0,2], 'g', s = 75)
ax.scatter(result_Mars.solution_CCI[0,0], result_Mars.solution_CCI[0,1], result_Mars.solution_CCI[0,2], 'k', s = 75)
ax.scatter(result_Jupiter.solution_CCI[0,0], result_Jupiter.solution_CCI[0,1], result_Jupiter.solution_CCI[0,2], 'orange', s = 75)
ax.scatter(0, 0, 0, s=100, color='r')
for i in range(0, len(result_3I.t), int(len(result_3I.t)/10) ):
    ax.scatter(result_3I.solution_CCI[i,0], result_3I.solution_CCI[i,1], result_3I.solution_CCI[i,2], s=10, color='b')
    ax.scatter(result_Earth.solution_CCI[i,0], result_Earth.solution_CCI[i,1], result_Earth.solution_CCI[i,2], s=10, color='g')
    ax.scatter(result_Mars.solution_CCI[i,0], result_Mars.solution_CCI[i,1], result_Mars.solution_CCI[i,2], s=10, color='k')
    ax.scatter(result_Jupiter.solution_CCI[i,0], result_Jupiter.solution_CCI[i,1], result_Jupiter.solution_CCI[i,2], s=10, color='orange')
    ax.plot( [result_3I.solution_CCI[i,0], result_Earth.solution_CCI[i,0]], [result_3I.solution_CCI[i,1], result_Earth.solution_CCI[i,1]], [result_3I.solution_CCI[i,2], result_Earth.solution_CCI[i,2]] , 'g', alpha=0.6 )
    ax.plot( [result_3I.solution_CCI[i,0], 0], [result_3I.solution_CCI[i,1], 0], [result_3I.solution_CCI[i,2], 0] , 'b', alpha=0.6 )
    #ax.plot( [result_3I.solution_CCI[i,0], result_Mars.solution_CCI[i,0]], [result_3I.solution_CCI[i,1], result_Mars.solution_CCI[i,1]], [result_3I.solution_CCI[i,2], result_Mars.solution_CCI[i,2]] , 'k', alpha=0.3 )
ax.set_xlim([-7*one_AU_in_m, 7*one_AU_in_m])
ax.set_ylim([-7*one_AU_in_m, 7*one_AU_in_m])
fig.set_facecolor('black')
ax.set_facecolor('black') 
plt.grid()

plt.figure()
plt.subplot(131)
plt.plot(result_3I.t, result_3I.magr/one_AU_in_m)
plt.xlabel("t (sec)")
plt.ylabel("3I distance from heliocenter (mAU)")
plt.grid()
plt.subplot(132)
plt.plot(result_3I.t, result_3I.magv/(10**3))
plt.xlabel("t (sec)")
plt.ylabel("3I velocity relative to heliocenter (km/s)")
plt.grid()
plt.subplot(133)
plt.plot(result_3I.t/one_day, result_3I.solution[:, -1], label="M")
plt.plot(result_3I.t/one_day, result_3I.true_anom, label="T")
plt.ylabel("anom (deg)")
plt.xlabel("t (days)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
