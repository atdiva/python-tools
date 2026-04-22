import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.spatial import distance 

'''
“You’re given a dataset of samples collected across a region. 
Each sample has measurements of multiple variables. 
In our setting, certain combinations of variables tend to co-occur due to underlying processes. 
We’re interested in identifying samples that deviate from the background behavior 
— these could indicate something unusual or interesting.”
'''

N = 1000 #num of measurements 
xsim = np.linspace(0, 2, N)
df = pd.DataFrame( {"A" : 0.5*xsim + 0.1 + np.random.normal(0, 0.1, N), 
                    "B" : 0.3*xsim + 0.2 + np.random.normal(0, 0.1, N), 
                    "C" : np.random.normal(0, 1, N) })

#univariate high A values
Astart = int(0.2*N)
Aend = int(0.3*N)
df.loc[Astart:Aend, "A"] = np.random.uniform(1, 3, Aend-Astart + 1 )*np.array(df["A"].loc[Astart:Aend])

#Correlation breakers: A high but B low (violates relationship)
df.loc[Astart:Aend, "B"] = np.random.normal(0, 0.1, Aend-Astart + 1)

#Multivariate anomolies
random_indx = np.random.randint(0, N, 50)
random_values = np.random.uniform(0, 0.5, len(random_indx))
df.loc[random_indx, "A"] = random_values
df.loc[random_indx, "B"] = random_values
df.loc[random_indx, "C"] = random_values

#Quick look
print(df.head())
print(df.describe())
print(np.sum(df.isnull()))

x = np.arange(0, len(df), 1)
plt.figure()
plt.subplot(121)
plt.scatter(x, df["A"], alpha=0.5, label="A")
plt.scatter(x, df["B"], alpha=0.5, label="B")
plt.scatter(x, df['C'], alpha=0.2, label="C")
plt.grid()
plt.legend()
plt.subplot(122)
plt.hist(df["A"], alpha=0.5, bins=30, label="A")
plt.hist(df["B"], alpha=0.5, bins=30, label="B")
plt.hist(df['C'], alpha=0.5, bins=30, label="C")
plt.grid()
plt.legend()


plt.figure()
plt.scatter(df['A'], df['B'])
plt.xlabel("A")
plt.ylabel("B")
plt.grid()

#proceeding with mahalonobis method because distributions look nearly gaussian
residuals = df['B'] 

plt.figure()
plt.scatter(df['A'], residuals, alpha=0.5)
plt.grid()

#chi-square corresponds to pvalue
chi_square_threshold = 2
data = np.array( [df['A'], residuals] ).T
mean = [ np.mean(data[:,0]), np.mean(data[:,1]) ]
covariance = np.cov(data[:,0], data[:,1] )
inv_cov = np.linalg.inv(covariance)

maha_distances = np.zeros([len(data), ])
for i in range(len(data)):
    maha_distances[i] = distance.mahalanobis( [data[i, 0], data[i, 1]], mean, inv_cov)

#indicies of outliers
anom_ind = (maha_distances**3 > 9.21) #using p value table of chi squared df = 2
plt.scatter(df['A'][anom_ind], residuals[anom_ind], marker='x', c='r', alpha=0.5)
plt.xlabel("A")
plt.ylabel("residuals")

plt.show()