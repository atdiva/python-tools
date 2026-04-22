import util as util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

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

def f_dynamics_only_g(t, pos_vel, params):

    #parameters to integrate
    vx = pos_vel[3]
    vy = pos_vel[4]
    vz = pos_vel[5]

    mu = params[0]

    gc = util.grav_acceleration(pos_vel[0:3], mu) 

    return np.array( [ vx, vy, vz, gc[0], gc[1], gc[2] ] )

def f_dynamics_with_g(t, pos_vel, params):

    if np.isnan(pos_vel).any():
        return np.nan*np.ones(6)

    #parameters to integrate
    x = pos_vel[0] 
    y = pos_vel[1]
    z = pos_vel[2]
    vx = pos_vel[3]
    vy = pos_vel[4]
    vz = pos_vel[5]

    mu = params[0]
    xf = params[1]
    yf = params[2]
    zf = params[3]
    dt = params[4]
    timetogo = params[5]
    zemdstop = params[6]
    
    #time left in intercept 
    timeleft = timetogo - t

    gc = util.grav_acceleration(pos_vel[0:3], mu) #current g vector
    xT_t2 = np.array([xf, yf, zf]) #stationary target, intercept point

    if timeleft <= 0.0:
        print("WARN: timeleft is <= 0")
        return np.array( [ vx, vy, vz, gc[0], gc[1], gc[2] ] )

    '''
    dt_eff = min(dt, timeleft)
    tvec = np.arange(0, timeleft + dt_eff, dt_eff)
    #a. cartesian integration rk4
    xO = util.rk4(tvec, f_dynamics_only_g, pos_vel, params)
    xO_t2 = xO[-1, 0:3]
    '''
    
    #b. using KO propagator 
    current_KO_state = util.OrbitalSimulation.compute_KO_from_CCI(pos_vel, mu)
    #print(current_KO_state)
    current_KO_state_dict = {'a (m)': current_KO_state[0], 
                             'e': current_KO_state[1], 
                             'i (deg)': current_KO_state[2], 
                             'RAAN (deg)': current_KO_state[3], 
                             'aop (deg)': current_KO_state[4], 
                             'M (deg)': current_KO_state[5]}
    
    res = util.OrbitalSimulation(timeleft, current_KO_state_dict, "", dt, False, "Sun")
    xO_t2 = res.solution_CCI[-1, 0:3]

    ZEMD_t2 = xT_t2 - xO_t2
    acc_command = 2*ZEMD_t2/(timeleft**2)

    print(timeleft, np.sqrt( np.dot(ZEMD_t2,ZEMD_t2) ), np.sqrt( np.dot(acc_command,acc_command) ) )
    if timeleft <= 0:
        print("WARN: time left <= 0")
        acc_command = 0.0*np.zeros(len(acc_command))
        return np.nan*np.ones(6)
    if np.sqrt( np.dot(ZEMD_t2, ZEMD_t2) ) <= zemdstop:
        print("INFO: reached ZEM stopping criterion")
        acc_command = 0.0*np.zeros(len(acc_command))
        return np.nan*np.ones(6)
    elif timeleft < 2*dt:
        print("WARN: reached stop time criterion")
        acc_command = 0.0*np.zeros(len(acc_command))
        return np.nan*np.ones(6)

    acc = acc_command + gc

    return np.array( [ vx, vy, vz, acc[0], acc[1], acc[2] ] )
    
def compute_optimized_orbital_path(orbital_result, desired_CCI_position, timetogo, dt ):

    GM_central = orbital_result.mu
    init_CCI_state = orbital_result.solution_CCI[0,:]

    t = np.arange(start=orbital_result.t[0], stop = timetogo + dt, step=dt) 
    optimal_trajectory_CCI = np.zeros([len(t), 6])
    optimal_trajectory_CCI[0,:] = init_CCI_state

    params = [GM_central, 
              desired_CCI_position[0], 
              desired_CCI_position[1], 
              desired_CCI_position[2], 
              dt, 
              timetogo,
              6371*1000]
    
    optimal_trajectory_CCI = util.rk4(t, f_dynamics_with_g, init_CCI_state, params)

    #check where any one of the states is nan and remove 
    idx = np.where( np.isnan(optimal_trajectory_CCI[:,0]))

    return t[0:idx[0][0]], optimal_trajectory_CCI[0:idx[0][0], :]
    

#############################################################################

#constants and duration 
one_AU_in_m = 149597870691 #m
lunar_distance = 1000*384399 #m 
one_day = 3600*(24 + 56/60) #sec 
duration = 250*one_day #from time stamp of KO state
Earth_day_for_intercept = 200
Earth_intercept_time_seconds = Earth_day_for_intercept*one_day
sim_dt = one_day/5

#read data, note error due to osculating elements with no model 
df = pd.read_excel("data/Astro/OrbitalElements.xlsx", sheet_name=2)

#Get Earth's position using JPL ephemeris preloaded (see get_earth_epehm.py)
indx_Earth = df[df['Name'] == "Earth"].index[0]
initial_state_Earth = set_initial_state(df, indx_Earth)
result_Earth = util.OrbitalSimulation(duration, initial_state_Earth, "", one_day/100, False, "Sun")

#Get 3I's Keplerian Orbit
indx_3I = df[ df['Name'] == "3I/ATLAS"].index[0]
initial_state_3I = set_initial_state(df, indx_3I)
result_3I = util.OrbitalSimulation(duration, initial_state_3I, "", sim_dt/20, False, "Sun")

#intercept point 
ind_of_intercept_Earth = util.find_index_of_nearest( result_Earth.t, Earth_intercept_time_seconds )
intercept_point = result_Earth.solution_CCI[ind_of_intercept_Earth, 0:3]

init_3I_CCI_state = result_3I.solution_CCI[0,:]
t_deltav, optimal_solution = compute_optimized_orbital_path(result_3I,
                                                           intercept_point, #intercept position
                                                           Earth_intercept_time_seconds, #time in sec
                                                           sim_dt) 
#position vector from heliocentric origin
optimal_magr = np.sqrt( optimal_solution[:,0]**2 +  optimal_solution[:,1]**2 + optimal_solution[:,2]**2 )
optimal_magv = np.sqrt( optimal_solution[:,3]**2 +  optimal_solution[:,4]**2 + optimal_solution[:,5]**2 )
optimal_a = np.gradient(optimal_magv, t_deltav) 
optimal_maga = np.abs(optimal_a)
dv = (t_deltav[1] - t_deltav[0])*np.cumsum(optimal_maga)

optimal_magr_Earth = np.sqrt( (optimal_solution[:,0] - intercept_point[0] )**2 +  (optimal_solution[:,1]- intercept_point[1] )**2 + (optimal_solution[:,2]- intercept_point[2] )**2 )
magr_Earth = np.sqrt( (result_3I.solution_CCI[:,0] - result_Earth.solution_CCI[:,0] )**2 +  (result_3I.solution_CCI[:,1]- result_Earth.solution_CCI[:,1] )**2 + (result_3I.solution_CCI[:,2] - result_Earth.solution_CCI[:,2] )**2 )

#############################################################################

print("---------------------------------------------")
print("Time to Perihelion (days from epoch) : " + str( (1/one_day)*result_3I.t[ util.find_index_of_nearest(result_3I.magr, min(result_3I.magr)) ] ) )
print("Perihelion Distance (AU) : " + str( (1/one_AU_in_m)*result_3I.magr[ util.find_index_of_nearest(result_3I.magr, min(result_3I.magr)) ] ) )
print("Max speed v (km/s) : " + str( (1/1000)*result_3I.magv[ util.find_index_of_nearest(result_3I.magv, max(result_3I.magv)) ] ) )

print("---------------------------------------------")
print("optimal trajectory sim ended at " + str(t_deltav[-1]/one_day) + " (days)" )
print("intercept position delta (km) " + str(optimal_magr_Earth[-3:]/lunar_distance) + " (lunar distance)" )
print("max delta v (km/s) " + str(np.max(np.abs(dv[:-1]))/1000) )

#Plot
plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(result_3I.solution_CCI[:,0], result_3I.solution_CCI[:,1], result_3I.solution_CCI[:,2], 'blue')
ax.plot3D(optimal_solution[:,0], optimal_solution[:,1], optimal_solution[:,2], 'r')
ax.plot3D(result_Earth.solution_CCI[:,0], result_Earth.solution_CCI[:,1], result_Earth.solution_CCI[:,2], 'green')
ax.scatter(result_3I.solution_CCI[0,0], result_3I.solution_CCI[0,1], result_3I.solution_CCI[0,2], 'b', s = 75)
ax.scatter(result_Earth.solution_CCI[0,0], result_Earth.solution_CCI[0,1], result_Earth.solution_CCI[0,2], 'g', s = 200)
ax.scatter(result_Earth.solution_CCI[ind_of_intercept_Earth,0], result_Earth.solution_CCI[ind_of_intercept_Earth,1], result_Earth.solution_CCI[ind_of_intercept_Earth,2], 'c', s = 200)

ax.scatter(0, 0, 0, s=1000, color='y')
for i in range(0, len(result_3I.t), int(len(result_3I.t)/10) ):
    ax.scatter(result_3I.solution_CCI[i,0], result_3I.solution_CCI[i,1], result_3I.solution_CCI[i,2], s=10, color='b')
    ax.plot( [result_3I.solution_CCI[i,0], 0], [result_3I.solution_CCI[i,1], 0], [result_3I.solution_CCI[i,2], 0] , 'b', alpha=0.6 )
    ax.scatter(result_Earth.solution_CCI[i,0], result_Earth.solution_CCI[i,1], result_Earth.solution_CCI[i,2], s=10, color='g')
    #ax.plot( [result_3I.solution_CCI[i,0], result_Earth.solution_CCI[i,0]], [result_3I.solution_CCI[i,1], result_Earth.solution_CCI[i,1]], [result_3I.solution_CCI[i,2], result_Earth.solution_CCI[i,2]] , 'g', alpha=0.6 )

for i in range(0, len(t_deltav), int(len(t_deltav)/10) ):
    ax.scatter(optimal_solution[i,0], optimal_solution[i,1], optimal_solution[i,2], s=10, color='r')
    #ax.plot( [optimal_solution[i,0], result_Earth.solution_CCI[i,0]], [optimal_solution[i,1], result_Earth.solution_CCI[i,1]], [optimal_solution[i,2], result_Earth.solution_CCI[i,2]] , 'r', alpha=0.6 )

ax.set_xlim([-4*one_AU_in_m, 4*one_AU_in_m])
ax.set_ylim([-4*one_AU_in_m, 4*one_AU_in_m])
fig.set_facecolor('black')
ax.set_facecolor('black') 
plt.grid()

plt.figure(dpi=150)
plt.subplot(2,3,1)
plt.plot(result_3I.t/one_day, result_3I.magr/one_AU_in_m, 'b', label="keplerian")
plt.plot(t_deltav[:-1]/one_day, optimal_magr[:-1]/one_AU_in_m, 'r', label="optimal")
plt.xlabel("t (days)")
plt.ylabel("3I distance from heliocenter (AU)")
plt.ylim([0, 4])
plt.legend()
#plt.tight_layout()
plt.grid()
plt.subplot(2,3,2)
plt.plot(result_3I.t/one_day, result_3I.magv/(10**3), 'b',  label="keplerian")
plt.plot(t_deltav[:-1]/one_day, optimal_magv[:-1]/(10**3), 'r',label="optimal")
plt.xlabel("t (days)")
plt.ylabel("3I velocity relative to heliocenter (km/s)")
plt.grid()
plt.legend()
#plt.tight_layout()
plt.subplot(2,3,3)
plt.plot(result_3I.t/one_day, result_3I.solution[:, -1], label="M")
plt.plot(result_3I.t/one_day, result_3I.true_anom, label="T")
plt.ylabel("anom (deg)")
plt.xlabel("t (days)")
plt.legend()
plt.grid()
#plt.tight_layout()
plt.subplot(2,3,4)
plt.plot(t_deltav[:-1]/one_day, optimal_magr[:-1]/one_AU_in_m, 'b', label='dist from Sun')
plt.plot(t_deltav[:-1]/one_day, optimal_magr_Earth[:-1]/one_AU_in_m, 'r', label='dist from Earth-3I/ATLAS intercept point')
plt.plot(result_Earth.t[:-1]/one_day, magr_Earth[:-1]/one_AU_in_m, 'k', label='dist from Earth (KO)')
plt.legend()
plt.ylabel("|r| ")
plt.xlabel("t (days)")
plt.grid()
#plt.tight_layout()
plt.subplot(2,3,5)
plt.plot(t_deltav[:-1], optimal_a[:-1], 'r')
plt.ylabel("a (m/sec2)")
plt.xlabel("t (sec)")
plt.grid()
#plt.tight_layout()
plt.subplot(2,3,6)
plt.plot(t_deltav[:-1], dv[:-1]/1000, 'r')
plt.ylabel("dv (km/s)")
plt.xlabel("t (sec)")
plt.grid()
#plt.tight_layout()
plt.show()
