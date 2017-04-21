import numpy as np

def normalizeData(X,mean,std):
    normalizedX = (X-mean)/std
    return normalizedX

def getMeanStd(X):
    Xmean = np.mean(X,0)
    Xstd = np.std(X,0)
    return Xmean, Xstd

def toFile(A,filename):
    file = open(filename,"w")
    for i in range(len(A)):
        file.write(A[i])

def yToMatrix(y,nOfOutput):
    #take input as y (m*1), return ys as one-hot vectors (m*10)
    m = y.shape[0]
    ys = np.zeros((m,nOfOutput))
    for i in range(m):
        ys[i,y[i]] = 1
    return ys

def divideDataSet(filename,nameBase,percentageForTrain=0.8,percentageForValidation=0.1):
    #filename should be the data set
    #make sure label is on the first column and there is no string
    #run this only once to separate data into training set, validation set and test set
    #nameBase is the data set's name without postfix
    #this will not do normalization
    dataFile = open(filename)
    lines = dataFile.readlines()
    np.random.shuffle(lines)
    m = len(lines)
    nTrain = int(m*percentageForTrain)
    nValidate = int(m*percentageForValidation)
    nTest = m - nTrain-nValidate
    dataTrain = lines[:nTrain]
    dataValidate =lines[nTrain:nTrain+nValidate]
    dataTest = lines[nTrain+nValidate:]
    toFile(dataTrain,nameBase+"_train.csv")
    toFile(dataValidate,nameBase+"_validate.csv")
    toFile(dataTest,nameBase+"_test.csv")
    print("Data divided, total data entry:",m,"n for training:",nTrain,"n for validating:",nValidate,"n for testing",nTest)