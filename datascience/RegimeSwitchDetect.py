import numpy as np 
import util as util 
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import signal
import pandas as pd
from scipy import optimize, signal
from alive_progress import alive_bar

#Helper functions of algo imported from utils 
def compute_log_return(data):
        output = np.zeros( np.shape(data) )
        output[0] = np.nan
        output[1:] = np.log( data[1:] / data[:-1] )
        return np.array(output)

def score(xs, index_switch):

    #split 
    left = xs[:index_switch]
    right = xs[index_switch:]
    #compute log return 
    log_left = compute_log_return(left)
    log_right = compute_log_return(right)

    #incase nans are in log return 
    mu_left = np.nanmean(log_left)
    mu_right = np.nanmean(log_right)
    sigma_left = np.nanstd(log_left)
    sigma_right = np.nanstd(log_right)

    metric = jsd_gaussian_approx(mu1=mu_left,  sigma1=sigma_left, 
                           mu2=mu_right, sigma2=sigma_right)
    
    return metric

def compute_hyperparameter(mu_sigma_distance):
    #empirical based
    distance = [0.69282, 0.59160, 0.489897, 0.38729, 0.28284, 0.0 ]
    alpha_vals = [ 12, 11.5, 11, 8.5, 8, 0]
    interp_alpha = np.interp(mu_sigma_distance, distance[::-1], alpha_vals[::-1])
    return interp_alpha

def logistic_curve(x, k, x0):
    return 1/(1 + np.exp(-k*(x - x0)) )

def hsfr(x, s):
    return  1.0*( x >= s ) 
def hsfl(x, s):
    return  1.0*( x < s ) 

#is zero when mu=mu and sigma=sigma
#https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
def kldiv(mu1, sigma1, mu2, sigma2):
    first_term = np.log(sigma2/sigma1)
    second_term = (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2)
    return first_term + second_term - 0.5

#Jensen-Shannon Divergence symmetric based on KL divergence (approximation using the first moment)
#normalized to a distance 
def jsd_gaussian_approx(mu1, sigma1, mu2, sigma2):
    mu_m = (1/2) * (mu1 + mu2)
    sigma_m = np.sqrt( (1/2)*(sigma1**2 + sigma2**2) + (1/4)*(mu1 - mu2)**2 )
    jsd = (1/2)*( kldiv(mu1, sigma1, mu_m, sigma_m) + kldiv(mu2, sigma2, mu_m, sigma_m) )
    jsd_distance = np.sqrt(jsd/np.log(2))

    return jsd_distance

def mu_sigma_distance(mu1, sigma1, mu2, sigma2):
    return np.abs(mu1 - mu2) + np.sqrt( np.abs( sigma1**2 - sigma2**2 ) )

def concat(a, b):
    return np.concatenate( (a, b), axis = 0)

def detect_identify_rsw(xs, given_index_switch):

    # hyper parameter dependent on ms_distance at given switch index
    log_left = compute_log_return(xs[:given_index_switch])
    log_right = compute_log_return(xs[given_index_switch:])
    mu_left = np.nanmean(log_left)
    mu_right = np.nanmean(log_right)
    sigma_left = np.nanstd(log_left)
    sigma_right = np.nanstd(log_right)
    given_mu_sigma_distance = mu_sigma_distance(mu_left, sigma_left, mu_right, sigma_right)

    #compute hyper parameter 
    alpha = compute_hyperparameter(given_mu_sigma_distance)
    
    #normalize (max = 1)
    xs = xs/np.max(xs)

    #compute metric(t) for each test point but first and last element
    #note this means metric_xs[0] and [-1] are 0.0
    metric_xs = np.nan*np.ones(len(xs))
    for i in range(1, len(metric_xs) - 1, 1): #skip first element and last element
        metric_xs[i] = score(xs, i)

    #replace nans/infs with 0.0
    metric_xs = np.nan_to_num(metric_xs, nan=0.0, posinf=0.0, neginf=0.0)
    #TODO: metric should not be greater than 1, so fix  
    #if np.max(metric_xs) > 1.0:
        #print("error, metric > 1")
    # any values greater than 1.0 replace with 0.0
    metric_xs[metric_xs > 1.0] = 0.0

    #signal processing prior to argmax(JSD(t))
    # 1: apply window to remove edge effects (this makes it difficult if the regime switch happens close to the edges)
    sig = metric_xs*signal.windows.tukey(len(metric_xs), alpha=0.02) #edge artifacts(limitation for edge switches)
    # 2: low frequency filter on sig(t)
    sig = signal.savgol_filter(sig, window_length=101, polyorder=1) 

    predicted_regime_switch_index = np.argmax(sig)

    #penalty for closeness of predicted verses given 
    penalty = np.exp(-alpha * (( given_index_switch - predicted_regime_switch_index) / len(xs))**2)

    #final metric
    if (predicted_regime_switch_index == 0 or predicted_regime_switch_index == len(xs) - 1) or (metric_xs[predicted_regime_switch_index] == 0.0) :
        
        print('error, predicted point is at the [0 or -1] or value at predicted point is 0')
        metric_term = sig[predicted_regime_switch_index]

        print(metric_xs[0], sig[0])
        print(metric_xs[-1], sig[-1])

        plt.figure()
        plt.plot(xs)
        plt.grid()
        plt.axvline(given_index_switch, c='r', alpha=0.7)
        plt.figure()
        plt.plot( sig, label='final', alpha =0.5)
        plt.plot( metric_xs, label='raw', alpha=0.5)
        plt.legend()
        plt.axvline(predicted_regime_switch_index, alpha=0.7)
        plt.axvline(given_index_switch, c='r', alpha=0.7)
        plt.axvline(np.argmax(sig), c='g', alpha=0.7)
        plt.tight_layout()
        plt.show()

    else:
        metric_term = metric_xs[predicted_regime_switch_index]

    metric = metric_term*penalty
    if metric == 0:
        print('error')
        print(metric_term, predicted_regime_switch_index, given_index_switch, penalty)
    
    #print(metric, penalty)
    #print("Given switch test point " + str(given_index_switch))
    #print("Predicted breakpoint " + str(predicted_regime_switch_index))
    #print("JSD at predicted " + str(metric_term))
    #print("Penatly term " + str(penalty))
    #print("Final Metric = " + str(metric_term*penalty))
    return metric, predicted_regime_switch_index, metric_term, penalty, metric_xs

############################################################################################################
def generate_regime_switch(num_samples):

    #regime distribution params
    mu_left = 0.0
    mu_right = mu_left
    min_sigma = 0.1
    max_sigma = 0.7 

    #flip for true injection or not
    if np.random.uniform(0, 1) <= 0.999999:
        true_injection = True

        #flip a coin for left/right side greater sigma
        if np.random.uniform(0,1) > 0.5:
            sigma_left = min_sigma
            sigma_right = max_sigma 
        else:
            sigma_left = max_sigma
            sigma_right = min_sigma

        switch_iteration = np.random.randint(num_samples*0.02, num_samples*0.98)

        #flip wether to specify the given point to be the true breaking point
        if np.random.uniform(0, 1) <= 0.5: #0.999999:
            guess_iteration = switch_iteration 
            switch = 1 #the given index is at the true location of the supplied boundary point
        else:
            guess_iteration = np.random.randint(1, num_samples)
            #ensure switch iteration is not close to guess iteration
            while ( np.abs(guess_iteration - switch_iteration) < 0.25*num_samples ):
                guess_iteration = np.random.randint(1, num_samples) 
            switch = 0 #means the given index is not at the location of the true existing regime switch point

        #left side rw
        rwbf = util.randomwalk(1, [mu_left, sigma_left], "normal", "geometric", switch_iteration )
        rwbf.sim_random_walk()
        nbf = rwbf.iteration
        rwbf_samples = rwbf.samples

        #right side rw
        rwaf = util.randomwalk(rwbf_samples[-1], [mu_right, sigma_right], "normal", "geometric", num_samples - switch_iteration  )
        rwaf.sim_random_walk()
        naf = nbf[-1] + rwaf.iteration
        rwaf_samples = rwaf.samples

        #combine 
        n = concat(nbf, naf)
        samples = concat(rwbf_samples, rwaf_samples)

    else:
        true_injection = False
        #no split
        rw = util.randomwalk(1, [ 0.5*(mu_right + mu_left), np.sqrt(sigma_right**2 + sigma_left**2)], "normal", "geometric", num_samples  )
        rw.sim_random_walk()
        n = rw.iteration
        samples = rw.samples
        guess_iteration = np.random.randint(1, num_samples) 
        switch = 0

    return n, samples, switch_iteration, guess_iteration, switch, mu_left, sigma_left, mu_right, sigma_right, true_injection

############################################################################################################

N = 3000 #size of time series 
M = 200 #number of time series 
df = pd.DataFrame(columns=["true_switch_point", "given_switch_point", "T/F", "metric", "predicted_break_point", "JSD", "penalty", "ms_distance", "injection"])
simdata = np.zeros([M, N])
with alive_bar(M) as bar:
    for i in range(M):
        n, x, tsw, gsw, tfsw, mul, sl, mur, sr, ti = generate_regime_switch(N)
        #print("True point " + str(tsw))
        mtr, pbk, jsd, pnlty, metric_time = detect_identify_rsw(x, gsw)
        simdata[i,:] = metric_time
        df.loc[len(df)] = [tsw, gsw, tfsw, mtr, pbk, jsd, pnlty, mu_sigma_distance(mul, sl, mur, sr), ti]
        bar()


#Logistic Plot 
true_df = df.loc[df['T/F'] == 1]
false_df = df.loc[df['T/F'] == 0]
#logistic fit 
popt, pcov = optimize.curve_fit(logistic_curve, df['metric'] , df['T/F'] )
xs_metric = np.linspace(0, 1, 2*M)
lcurve_fit = logistic_curve(xs_metric, *popt)

ind_true_negative = true_df['metric'] <= popt[1]
ind_false_positive  = false_df['metric'] > popt[1]
print("INFO: Logistic Fit Values: " + str(popt))
print("INFO: True Negative percent " + str( 100*len( true_df[ind_true_negative])/len(true_df) )  )
print(true_df[ind_true_negative])
print("INFO: False Positive percent " + str( 100*len( false_df[ind_false_positive])/len(false_df) )  )
print(false_df[ind_false_positive])
sorted_metrics_false = np.sort(false_df['metric'])
print("Highest True Negative metric = " + str( sorted_metrics_false[ -4: ][::-1] ) )

plt.close("all")
plt.figure()
plt.subplot(131)
plt.scatter(true_df['metric'], true_df['T/F'], c='r')
plt.scatter(false_df['metric'], false_df['T/F'], c='k')
plt.plot(xs_metric, lcurve_fit, 'g')
plt.grid()
plt.xlabel("metric")
plt.ylabel("True/False switch")
plt.subplot(132)
sns.histplot(true_df['ms_distance'],  label="T")
sns.histplot(false_df['ms_distance'], label="F")
plt.legend()
plt.grid()
plt.xlabel("MS distance")
plt.subplot(133)
#sns.histplot(true_df['JSD'], binwidth=0.05, label='T JSD')
#sns.histplot(false_df['JSD'], binwidth=0.05, label="F JSD")
sns.histplot(true_df['metric'], binwidth=0.05, label='T m')
sns.histplot(false_df['metric'], binwidth=0.05, label='F m')
plt.xlabel("score")
plt.axvline(0.5, c='k')
plt.legend()
plt.grid()

#of the ones where given index is the true break point
percents = np.linspace(0, 1, 1000)
percent_metric = np.zeros(len(percents))
percent_metric = [ len(true_df[true_df['metric'] <= per])/len(true_df) for per in percents ]
plt.figure()
plt.plot(percents, percent_metric)
plt.xlabel("percent of")
plt.ylabel("percent metric (true) ")
plt.grid()
plt.show()

decbound = popt[1]
print("Decision boundary LR at metric score = " + str(decbound) )
print("False Positives % : "  +  str(100*len(false_df[false_df['metric'] > decbound ])/len(false_df)))
print("True Negatives % : "  + str(100*len(true_df[true_df['metric'] <= decbound ])/len(true_df)))

######################################################

'''
#KL vs MS Distance
plt.close("all")
plt.figure()
sns.jointplot(df, x="ms_distance", y="kl")
plt.grid()
plt.tight_layout()
plt.show()
'''

'''
#Plot of an example
plt.close('all')
plt.figure()
plt.title( "metric at input split = " + str(mtr) )
plt.subplot(121)
plt.plot(n, x)
plt.vlines(tsw, min(x), max(x), 'r', label='true')
plt.vlines(gsw, min(x), max(x), 'g', label='input')
plt.grid()
plt.legend()
plt.xlabel("sample number")
plt.ylabel('value')
plt.tight_layout()
plt.subplot(122)
sns.histplot(log_l, kde=True, binwidth=0.001, color='blue', stat="percent", label="left")
sns.histplot(log_r, kde=True, binwidth=0.001, color='green', stat="percent", label="right")
plt.axvline(np.mean(log_l), color='b')
plt.axvline(np.mean(log_r), color='g')
plt.legend()
plt.xlabel("log returns")
plt.grid()
plt.tight_layout()
plt.show()
'''