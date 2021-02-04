"""
Atif Ahmad
CS 177
October 26, 2017
HW #3
"""
# Import numpy and make floating point division the default for Python 2.x
from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt

# Load data 
S = np.load('eruptions.npy')  # vector of observed eruption times
T = np.load('waiting.npy')    # vector of observed waiting times
n = S.shape[0]                # number of observations

# Plot data
plt.plot(S, T, 'ok')
plt.xlabel('Eruption Time (minutes)')
plt.ylabel('Waiting Time to Next Eruption (minutes)')
plt.show()

# Compute mean under empirical distribution
meanS = np.sum(S)/n
meanT = np.sum(T)/n

#4(a.)
sum = 0
for i in range(0, n):
    value = S[i] * S[i]
    sum = sum + value
    
expectSSquared = sum/n
meanSSquared = meanS * meanS

VarS = expectSSquared - meanSSquared
 
sum = 0
for i in range(0, n):
    value = T[i] * T[i]
    sum = sum + value
    
expectTSquared = sum/n
meanTSquared = meanT * meanT

VarT = expectTSquared - meanTSquared

print 'Var[S] = ', VarS
print 'Var[T] = ', VarT

#4.(b.)
minIndex = np.argmin(S)
s1bar = 0.25 * n - 1 + S[minIndex]
s1bar = np.floor(s1bar)
s2bar = 0.50 * n - 1 + S[minIndex]
s2bar = np.floor(s2bar)
s3bar = 0.75 * n - 1 + S[minIndex]
s3bar = np.floor(s3bar)

minIndex = np.argmin(T)
t1bar = 0.25 * n - 1 + T[minIndex]
t1bar = np.floor(t1bar)
t2bar = 0.50 * n - 1 + T[minIndex]
t2bar = np.floor(t2bar)
t3bar = 0.75 * n - 1 + T[minIndex]
t3bar = np.floor(t3bar)

print 's1bar = ', s1bar
print 's2bar = ', s2bar
print 's3bar = ', s3bar
print 't1bar = ', t1bar
print 't2bar = ', t2bar
print 't3bar = ', t3bar

#4.(c.)
sum = 0
for i in range (0,n):
    if (S[i] <= 3.5 and T[i] <= 70):
        sum = sum + S[i] + T[i]
    if (S[i] > 3.5 and T[i] > 70):
        sum = sum + S[i] + T[i]

pXY = sum / (np.sum(S) + np.sum(T))
pX = np.sum(S) / (np.sum(S) + np.sum(T))
pY = np.sum(T) / (np.sum(S) + np.sum(T))

print 'pXY(x,y) = ', pXY
print 'pX(x) = ', pX
print 'pY(y) = ', pY

#4.(d.)
checkValue = pX * pY
print 'pX(x)*pY(y) = ', checkValue

# Thresholds used to define X,Y variables in parts (c,d)
threshX = 3.5
threshY = 70