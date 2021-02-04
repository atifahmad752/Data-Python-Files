"""
Atif Ahmad
CS 177
October 16, 2017
HW #2
"""

#2.(a.)
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
         [5.0/6,(5.0/6)**2,(5.0/6)**3,(5.0/6)**4,(5.0/6)**5,(5.0/6)**6,
          (5.0/6)**7,(5.0/6)**8,(5.0/6)**9,(5.0/6)**10,(5.0/6)**11,
          (5.0/6)**12,(5.0/6)**13,(5.0/6)**14,(5.0/6)**15,(5.0/6)**16,
          (5.0/6)**17,(5.0/6)**18,(5.0/6)**19,(5.0/6)**20], 'ro')
plt.axis([1, 20, 0, 1])
plt.show()

def load_data():
    data = loadmat('enron.mat')
    trainFeat = np.array(data['trainFeat'], dtype=bool)
    trainLabels = np.squeeze(data['trainLabels'])
    testFeat = np.array(data['testFeat'], dtype=bool)
    testLabels = np.squeeze(data['testLabels'])
    vocab = np.squeeze(data['vocab'])
    vocab = [vocab[i][0].encode('ascii', 'ignore') for i in xrange(len(vocab))]
    data = dict(trainFeat=trainFeat, trainLabels=trainLabels,
                testFeat=testFeat, testLabels=testLabels, vocab=vocab)
    return data

# Load data
data = load_data()
trainFeat = data['trainFeat']
trainLabels = data['trainLabels']
testFeat = data['testFeat']
testLabels = data['testLabels']
vocab = data['vocab']
W = len(vocab)
'''
    Data description:
    - trainFeat: (Dtrain, W) logical 2d-array of word appearance for training documents.
    - trainLabels: (Dtrain,) 1d-array of {0,1} training labels where 0=ham, 1=spam.
    - testFeat: (Dtest, W) logical 2d-array of word appearance for test documents.
    - testLabels:  (Dtest,) 1d-array of {0,1} test labels where 0=ham, 1=spam.
    - vocab: (W,) 1d-array where vocab[i] is the English characters for word i.
'''

# Different possible vocabularies to use in classification, uncomment chosen line
vocabInds =  179  # Part (c): "money"
# vocabInds =  859  # Part (d): "thanks"
# vocabInds = 2211  # Part (e): "possibilities"
# vocabInds = [179, 859, 2211]  # Part (f): "money", "thanks", & "possibilities"
vocabInds = np.arange(W)  # Part (g): full vocabularly of all W words

# Separate "ham" and "spam" classes, subsample selected vocabulary words
trainHam  = trainFeat[trainLabels == 0][:, vocabInds]
trainSpam = trainFeat[trainLabels == 1][:, vocabInds]

# Number of training examples of each class
numHam = len(trainHam)
numSpam = len(trainSpam)

#4.(b)
# Count number of times each word occurs in each class
countsHam = np.sum(trainHam, axis=0)
# P(X_ij=1 | Y_i=H) can be computed from countsHam and numHam
PX1YH = numHam / countsHam
countsSpam = np.sum(trainSpam, axis=0)
# P(X_ij=1 | Y_i=S) can be computed from countsSpam and numSpam
PX1YS = countsSpam / numSpam

PX0YS = (numSpam - countsSpam)/ numSpam
PX0YH = (numHam - countsHam) / numHam

#4.(c)
print 'PX1YS = ', PX1YS
print 'PX1YH = ', PX1YH

if((PX0YS + PX1YS) > (PX0YH + PX1YH)):
    Ahati = 1
else:
    Ahati = 0

sum = 0
for i in range(1, vocabInds + 1):
    errorRate = (1 /vocabInds)*abs(Ahati - (PX0YS + PX1YS +PX0YH + PX1YH))
    sum = sum + errorRate
        
errorRate = sum
accuracy = 1 - errorRate
print 'Accuracy = ', accuracy

#Problem 4.(d.)
vocabInds =  859  # Part (d): "thanks"
# vocabInds = 2211  # Part (e): "possibilities"
# vocabInds = [179, 859, 2211]  # Part (f): "money", "thanks", & "possibilities"
vocabInds = np.arange(W)  # Part (g): full vocabularly of all W words

# Separate "ham" and "spam" classes, subsample selected vocabulary words
trainHam  = trainFeat[trainLabels == 0][:, vocabInds]
trainSpam = trainFeat[trainLabels == 1][:, vocabInds]

# Number of training examples of each class
numHam = len(trainHam)
numSpam = len(trainSpam)

# Count number of times each word occurs in each class
countsHam = np.sum(trainHam, axis=0)
# P(X_ij=1 | Y_i=H) can be computed from countsHam and numHam
PX1YH = numHam / countsHam
countsSpam = np.sum(trainSpam, axis=0)
# P(X_ij=1 | Y_i=S) can be computed from countsSpam and numSpam
PX1YS = countsSpam / numSpam

PX0YS = (numSpam - countsSpam)/ numSpam
PX0YH = (numHam - countsHam) / numHam

print 'PX1YS = ', PX1YS
print 'PX1YH = ', PX1YH

if((PX0YS + PX1YS) > (PX0YH + PX1YH)):
    Ahati = 1
else:
    Ahati = 0

sum = 0
for i in range(1, vocabInds + 1):
    errorRate = (1 /vocabInds)*abs(Ahati - (PX0YS + PX1YS +PX0YH + PX1YH))
    sum = sum + errorRate
        
errorRate = sum
accuracy = 1 - errorRate
print 'Accuracy = ', accuracy

#Problem 4.(e.)
vocabInds = 2211  # Part (e): "possibilities"
# vocabInds = [179, 859, 2211]  # Part (f): "money", "thanks", & "possibilities"
vocabInds = np.arange(W)  # Part (g): full vocabularly of all W words

# Separate "ham" and "spam" classes, subsample selected vocabulary words
trainHam  = trainFeat[trainLabels == 0][:, vocabInds]
trainSpam = trainFeat[trainLabels == 1][:, vocabInds]

# Number of training examples of each class
numHam = len(trainHam)
numSpam = len(trainSpam)

# Count number of times each word occurs in each class
countsHam = np.sum(trainHam, axis=0)
# P(X_ij=1 | Y_i=H) can be computed from countsHam and numHam
PX1YH = numHam / countsHam
countsSpam = np.sum(trainSpam, axis=0)
# P(X_ij=1 | Y_i=S) can be computed from countsSpam and numSpam
PX1YS = countsSpam / numSpam

PX0YS = (numSpam - countsSpam)/ numSpam
PX0YH = (numHam - countsHam) / numHam

print 'PX1YS = ', PX1YS
print 'PX1YH = ', PX1YH

if((PX0YS + PX1YS) > (PX0YH + PX1YH)):
    Ahati = 1
else:
    Ahati = 0

sum = 0
for i in range(1, vocabInds + 1):
    errorRate = (1 /vocabInds)*abs(Ahati - (PX0YS + PX1YS +PX0YH + PX1YH))
    sum = sum + errorRate
        
errorRate = sum
accuracy = 1 - errorRate
print 'Accuracy = ', accuracy

#4.(f.)

vocabInds = [179, 859, 2211]  # Part (f): "money", "thanks", & "possibilities"
vocabInds = np.arange(W)  # Part (g): full vocabularly of all W words

# Separate "ham" and "spam" classes, subsample selected vocabulary words
trainHam  = trainFeat[trainLabels == 0][:, vocabInds]
trainSpam = trainFeat[trainLabels == 1][:, vocabInds]

# Number of training examples of each class
numHam = len(trainHam)
numSpam = len(trainSpam)

# Count number of times each word occurs in each class
countsHam = np.sum(trainHam, axis=0)
# P(X_ij=1 | Y_i=H) can be computed from countsHam and numHam
PX1YH = numHam / countsHam
countsSpam = np.sum(trainSpam, axis=0)
# P(X_ij=1 | Y_i=S) can be computed from countsSpam and numSpam
PX1YS = countsSpam / numSpam

PX0YS = (numSpam - countsSpam)/ numSpam
PX0YH = (numHam - countsHam) / numHam

if((PX0YS + PX1YS) > (PX0YH + PX1YH)):
    Ahati = 1
else:
    Ahati = 0

sum = 0
for i in range(1, vocabInds + 1):
    errorRate = (1 /vocabInds)*abs(Ahati - (PX0YS + PX1YS +PX0YH + PX1YH))
    sum = sum + errorRate
        
errorRate = sum
accuracy = 1 - errorRate
print 'Accuracy = ', accuracy

#4.(g.)

vocabInds = np.arange(W)  # Part (g): full vocabularly of all W words

# Separate "ham" and "spam" classes, subsample selected vocabulary words
trainHam  = trainFeat[trainLabels == 0][:, vocabInds]
trainSpam = trainFeat[trainLabels == 1][:, vocabInds]

# Number of training examples of each class
numHam = len(trainHam)
numSpam = len(trainSpam)

# Count number of times each word occurs in each class
countsHam = np.sum(trainHam, axis=0)
# P(X_ij=1 | Y_i=H) can be computed from countsHam and numHam
PX1YH = np.log(numHam / countsHam)
countsSpam = np.sum(trainSpam, axis=0)
# P(X_ij=1 | Y_i=S) can be computed from countsSpam and numSpam
PX1YS = np.log(countsSpam / numSpam)

PX0YS = np.log((numSpam - countsSpam)/ numSpam)
PX0YH = np.log((numHam - countsHam) / numHam)

if((PX0YS + PX1YS) > (PX0YH + PX1YH)):
    Ahati = 1
else:
    Ahati = 0

sum = 0
for i in range(1, vocabInds + 1):
    errorRate = (1 /vocabInds)*abs(Ahati - (PX0YS + PX1YS +PX0YH + PX1YH))
    sum = sum + errorRate
        
errorRate = sum
accuracy = 1 - errorRate
print 'Accuracy = ', accuracy

# Display words that are common in one class, but rare in the other
ind = np.argsort(countsHam-countsSpam)
print 'Words common in Ham but not Spam:'
for i in xrange(-1, -100, -1):
    print vocab[ind[i]],
print
print 'Words common in Spam but not Ham:'
for i in xrange(100):
    print vocab[ind[i]],
