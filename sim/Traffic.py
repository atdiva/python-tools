import numpy as np 
import matplotlib.pyplot as plt
import util as util
import matplotlib.cm as cm
import random

#global
dt = 0.01
Npart = 30
duration = 60
max_road_length = 50
desired_seperation = 3
adjustment_velocity = 0.5
init_vel = 5
max_speed_limit = 8
min_speed_limit = 2

def road(x):
    return 10*np.sin(x)
def droad(x):
    return 10*np.cos(x)
def arclength_along_curve(pos_xy_one, pos_xy_two):
    #tested and checked with 2 particles, init pos [0, pi] against analytical value
    xs = np.linspace(pos_xy_one[0], pos_xy_two[0], 20)
    return np.trapz( np.sqrt(1 + droad(xs)**2 ) , xs )

def unit_tangent_vector_at_x(x):
    tang = [1, droad(x)]
    unit_tang = tang/np.linalg.norm(tang)
    return unit_tang

def advance_state( previous_time_states_of_all_particles , dt, jth_particle):

    state = previous_time_states_of_all_particles[:, jth_particle]
    x = state[0]
    #y = state[1]
    vx = state[2]
    vy = state[3]
    vmag = np.sqrt(vx**2 + vy**2)
    utang_vec = unit_tangent_vector_at_x(x)

    #if last particle just propagate forward
    if j is Npart - 1: 
        vnew = vmag*(utang_vec)
        new_state = [state[0] + dt*vnew[0], state[1] + dt*vnew[1], vnew[0], vnew[1]]
        return new_state
    else: 
        #check arc length distance between this and the particle in front
        distance_between_jth_and_in_front = arclength_along_curve(state[0:2], previous_time_states_of_all_particles[ 0:2, j+1] )
        if distance_between_jth_and_in_front == desired_seperation: #proceed as usual
            vnew = vmag*(utang_vec)
        elif distance_between_jth_and_in_front > desired_seperation: #speed up until max speed limit
            speed_up_to_vel = vmag + adjustment_velocity
            if speed_up_to_vel > max_speed_limit:
                vnew = vmag*(utang_vec) #can't speed past max vel
            else:
                vnew = (vmag + adjustment_velocity)*(utang_vec)
        else: #slow down 
            slow_down_to_vel = vmag - adjustment_velocity
            if slow_down_to_vel < min_speed_limit:
                vnew = vmag*(utang_vec) #cant slow down past min velocity 
            else:
                vnew = (vmag - adjustment_velocity)*(utang_vec)
            
        new_state = [state[0] + dt*vnew[0], state[1] + dt*vnew[1], vnew[0], vnew[1]]
        return new_state

#Design Road (single lane)
t = np.arange(0, duration, dt)
x = np.linspace(0, max_road_length, 1000) 
curve = road(x)



#1) equal spereation with pi spacing 
#init_x_pos = [0, np.pi]
#2) random positions
init_x_pos = np.sort(np.random.uniform(0, 0.1*max_road_length, Npart))
#init_x_pos = np.linspace(0, x[-1]/200, Npart)

#[t, [x, y, vx, vy], nth particle]
all_states = np.zeros([len(t), 4, Npart])
all_states[0, 0, :] = init_x_pos
all_states[0, 1, :] = road(init_x_pos)

#set init pos/vel
all_states[0, 2, :] = np.ones(np.shape(init_x_pos))
all_states[0, 3, :] = droad(init_x_pos)
init_velocities = np.sqrt( all_states[0, 2, :]**2  + all_states[0, 3, :]**2 )
all_states[0, 2, :] = init_vel*(all_states[0, 2, :]/init_velocities)
all_states[0, 3, :] = init_vel*(all_states[0, 3, :]/init_velocities)

colors = cm.rainbow(np.random.uniform(0, 1, Npart))
fig, ax = plt.subplots(dpi=300 )
scat = ax.scatter(all_states[0,0,:], all_states[0,1,:], c=colors, s=20)
ax.plot(x, curve, 'k', 10)
ax.set_aspect("equal")
ax.grid()
plt.tight_layout()
updated_positions = np.zeros([Npart, 2])
for step in range(1, len(t)):
    #loops over left most particle to the right
    for j in range(Npart):
        #advance step of jth particle at this time
        newstate = advance_state(all_states[step-1, :, :], dt, j)
        #store
        all_states[step, :, j] = newstate
        updated_positions[j, :] = newstate[0:2]

    scat.set_offsets(updated_positions)
    ax.set_title(f"t = {step * dt:.2f}")
    plt.pause(dt)          # controls frame rate
    #print(updated_positions)
plt.show()