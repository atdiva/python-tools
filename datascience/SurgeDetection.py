import numpy as np 
import matplotlib.pyplot as plt

#surge model
def surge_model(t, tpeak, A):
    if np.isscalar(t):
        t = np.array([t])

    x_t = np.zeros(len(t))
    x_t[t < tpeak] = t[t < tpeak]**2 
    ind_between = (t >= tpeak) & (t <= 2*tpeak)
    x_t[ind_between] = ( t[ind_between]  - 2*tpeak)**2 
    x_t = A*x_t/(tpeak**2)
    print(np.max(x_t))
    return x_t
    
#states are time-invariant 
def stm(t, x):
    return np.eye(len(x), len(x))

#nonlinear measurement mapping matrix
def H(t, x, measdim):
    Hout = np.zeros((measdim, len(x) ))
    tpeak = x[0]
    Apeak = x[1]

    dhdtpeak = np.nan
    dhdApeak = np.nan
    if t < tpeak:
        dhdApeak = (t/tpeak)**2
        dhdtpeak = Apeak*(t**2)*(-2/(tpeak**3))
    elif (t >= tpeak) & (t <= 2*tpeak):
        dhdApeak = (1/tpeak**2)*(t - 2*tpeak)**2
        dhdtpeak = (-2*Apeak/(tpeak**3))*(t - 2*tpeak)**2 - (4*Apeak/(tpeak**2))*(t - 2*tpeak)
    else:
        dhdtpeak = 0
        dhdApeak = 0

    Hout[0, 0] = dhdtpeak
    Hout[0, 1] = dhdApeak
    return Hout

#input params 
tpeak_true = 200
A_true = 1.5
statedim = 2
mdim = 1

N = 1000
time = np.linspace(0, 500, N )
noise = np.random.normal(0, 1, N)
signal = surge_model(time, tpeak_true, A_true)
data = signal + noise

state = np.zeros([N, statedim])
covar = np.zeros([N, statedim, statedim])
state_signal = np.zeros(N)

#init state [tpeak, Amplitude] and covar
state[0,0] = tpeak_true*np.random.uniform(0.85, 1.25)
state[0,1] = A_true*np.random.uniform(0.85, 1.25)
covar[0, 0, 0] = (50)**2
covar[0, 1, 1] = (1)**2

#R matrix
R = np.var(noise)

iden = np.eye(statedim, statedim)
#run loop
for i in range(1, len(time)):
    previous_state = np.reshape(state[i-1, :], (statedim, 1))
    previous_covariance = np.reshape(covar[i-1, : ,:], (statedim,statedim))
    stm_indx = stm(time[i], state[i-1])
    x_prop = np.dot(stm_indx, previous_state)
    P_prop = np.dot(stm_indx,  np.dot(previous_covariance, stm_indx.T) )

    #residual
    residual = data[i] - surge_model(time[i], x_prop[0], x_prop[1])

    #jacobian measurement mapping
    Hmat = H(time[i], x_prop, mdim)

    #compute kalman gain
    HPHT = np.dot(Hmat, np.dot(P_prop, Hmat.T))
    S = HPHT + R
    K = np.dot(P_prop, Hmat.T)/S

    #useful
    IminusKH = iden - np.dot(K, Hmat)
    KRKT = np.dot(K, np.dot(R, K.T))

    #update state
    x_update = x_prop + np.reshape( np.dot(K, residual), np.shape(x_prop) )
    P_update = np.dot(IminusKH, np.dot(P_prop, IminusKH.T)) + KRKT

    state[i, :] = np.reshape(x_update, (statedim, ))
    covar[i, :, :] = np.reshape(P_update, (statedim, statedim))
    state_signal[i] = surge_model(time[i], x_update[0], x_update[1])

#sigma's 
sigma_tpeak = np.sqrt(covar[:, 0, 0])
sigma_Apeak = np.sqrt(covar[:, 1, 1])

plt.figure()
plt.plot(time, data, 'k', alpha=0.8)
plt.plot(time, signal, 'r', alpha=0.8)
plt.plot(time, state_signal, 'b')

plt.figure()
plt.subplot(121)
plt.plot(time, state[:,0], 'b')
plt.plot(time, state[:,0] + 3*sigma_tpeak, '-b', linestyle='dashed')
plt.plot(time, state[:,0] - 3*sigma_tpeak, '-b', linestyle='dashed')
plt.axhline(tpeak_true, c='r')
plt.xlabel("time (sec)")
plt.ylabel("tpeak (sec)")
plt.grid()
plt.subplot(122)
plt.plot(time, state[:,1], 'b')
plt.plot(time, state[:,1] + 3*sigma_Apeak, 'b', linestyle='dashed')
plt.plot(time, state[:,1] - 3*sigma_Apeak, '-b', linestyle='dashed')
plt.axhline(A_true, c='r')
plt.xlabel("time (sec)")
plt.ylabel("A ")
plt.grid()
plt.show()