import copy
from math import exp, sqrt
from cmath import nan
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from random import randrange
import pandas as pd

#Create a DataFrame of train data
train = pd.DataFrame(pd.read_csv("train.csv", names = ['Id','MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape',
'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual',
'OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual',
'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF',
'TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces',
'FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive',
'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold',
'YrSold','SaleType','SaleCondition', 'SalePrice']))

#Create a DataFrame of test data
test = pd.DataFrame(pd.read_csv("test.csv", names = ['Id','MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape',
'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual',
'OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual',
'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF',
'TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces',
'FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive',
'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold',
'YrSold','SaleType','SaleCondition']))

#Drop columns made of string values in both data sets
train = train.drop(columns=['Id','MSZoning','Street','Alley','LotShape','LandContour','Utilities',
'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',
'MiscFeature','SaleType','SaleCondition'])

test = test.drop(columns=['Id','MSZoning','Street','Alley','LotShape','LandContour','Utilities',
'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',
'MiscFeature','SaleType','SaleCondition'])

#Predict binary classification using logistic function
def Prediction(row, weight):
    yhat = weight[-1]
    
    for i in range(len(row)-1):
        #print(row)
        yhat += row[i] * weight[i]

    yhat = (1.0 / (1.0 + exp(-yhat)))

    #print(yhat)
    return yhat

#Use stochastic gradient descent to get regression coefficients
def Weights(train, learnrate, epochs):
    weights = [0.0 for i in range(len(train[0]))]
    
    for epoch in range(epochs):
        sum_error = 0
        
        for row in train:
            yhat = Prediction(row, weights)
            error = row[-1] - yhat
            sum_error += error**2
            weights[-1] = weights[-1] + learnrate * \
            error * yhat * (1.0 - yhat)
            
            for i in range(len(row)-1):
                weights[i] = weights[i] + learnrate * error * \
                yhat * (1.0 - yhat) * row[i]
        #print('>epoch=%d, lrate=%.3f, error=%.3f' 
        #      % (epoch, l_rate, sum_error))
    return weights

#Find the min and max values for each column for later use in normalizing the dataset
def MinMax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

#Get min max values to denormalize sale prices
def SaleMinMax(saleprices):
    saleminmax = list()
    highest = 0
    lowest = 1000000
    
    for i in range(len(saleprices)):
        
        if saleprices[i] > highest:
            highest = saleprices[i]
        if saleprices[i] < lowest:
            lowest = saleprices[i]
    
    saleminmax.append(highest)
    saleminmax.append(lowest)

    return saleminmax
 
#Rescale dataset columns to the range 0-1 in order to have usable float numbers for error calculations
def Normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

#Return normalized values back to what they would have normally been to actually see price values
def Denormalize(dataset, saleminmax):
    for i in range(len(dataset)):
        dataset[i] = (dataset[i] * (saleminmax[0] - saleminmax[1]) + saleminmax[1])

#For algorithm validation, we run the validation set against the training set which is a version of the training set without saleprices
def LogisticRegression(train, test, learnrate, epochs):
    predictions = list()
    weights = Weights(train, learnrate, epochs)
    
    for row in test:
        yhat = Prediction(row, weights)
        #print(yhat)
        predictions.append(yhat)
    #print(predictions)

    return predictions

#Root Mean Squared Error Function
def RMSE(predictions, actual):
    rmse = 0
    
    for i in range(len(predictions)):
       rmse += sqrt(((predictions[i] - actual[i]) ** 2))
    
    rmse = rmse / len(predictions)

    return rmse

#Drop the first row of both datasets to remove the column titles from import
test = test.drop(labels = 0, axis = 0)
train = train.drop(labels = 0, axis = 0)

#Create a list of the column names left after purging ones we do not want
names = list(train)

#Change values of each dataset from int to float
for i in range(len(names)):
    train = train.astype({names[i]: 'float'})
    train = train.fillna(0) #make nan values 0
for i in range(len(names)-1):
    test = test.astype({names[i]: 'float'})
    test = test.fillna(0) #make nan values 0

#Create a list of the actual sale prices, then drop that column and form the validate data to get our accuracy later
saleprices = train['SalePrice'].values.tolist() #list of the column of saleprices for later reference
validate = train.drop(labels='SalePrice', axis=1) #validation set created from training data, removed the saleprice to allow for testing
train.drop(columns=['SalePrice'])

#Create a list of lists where each list corresponds to a row from the dataframe
testlist = test.values.tolist()
trainlist = train.values.tolist()
validatelist = validate.values.tolist()

#Get a minmax value for later use
minmax = MinMax(trainlist)
saleminmax = SaleMinMax(saleprices)

#Normalize the datasets so that they don't break float restrictions when calculating error
testnorm = copy.deepcopy(testlist)
trainnorm = copy.deepcopy(trainlist)
validatenorm = copy.deepcopy(validatelist)
Normalize(testnorm, minmax)
Normalize(trainnorm, minmax)
Normalize(validatenorm, minmax)

#Set the learn rate and the number of epochs we want to run our weight trainer, then run it to get a weight list
lr = 0.001
e = 50
#w = Weights(trainnorm, lr, e)

#Run LinearRegression using the validatelist against the trainlist to validate the accuracy of the algorithm
#print("Validation")
pd.set_option('display.max_rows', None) #set to print all lines of the dataframe
#print(trainnorm[-1])
#print(validatenorm[-1])
#print(w)
predictions = LogisticRegression(trainnorm, validatenorm, lr, e)
Denormalize(predictions, saleminmax)
avgrmse = RMSE(predictions, saleprices)
print("Average Error calculated from Validation Set")
print("avg RMSE = ", avgrmse)
print()
print("Validation")

data = {"Predictions": predictions, "SalePrices": saleprices}
df = pd.DataFrame(data)
print("Printed to Validate.txt file to view full set effectively")
#print(df)

#Determine the correctness of a prediction
predclassify = list()
priceclassify = list()

for i in range(len(predictions)):
    predictions[i] = round(predictions[i]) 
    if(predictions[i] > 180000):
        predclassify.append(1)
    else:
        predclassify.append(0)

for i in range(len(saleprices)):
    if(saleprices[i] > 180000):
        priceclassify.append(1)
    else:
        priceclassify.append(0)

#Create a confusion matrix using the data
cm = confusion_matrix(priceclassify, predclassify)
print('Confusion Matrix:\n',cm)
print(classification_report(priceclassify, predclassify))

#Create a text file output to view the full set
with open('LogValidation.txt', 'w') as f:
    f.write(df.to_string(header=['Prediction', 'Actual'], index=False))

#Run LinearRegression using the testlist against the trainlist to make predictions of house value on testlist homes
#print("Test")
predictions = LogisticRegression(trainnorm, testnorm, lr, e)
print()
print("Test")
Denormalize(predictions, saleminmax)
data = {"Predictions": predictions}
df = pd.DataFrame(data)
print("Printed to Test.txt file to view full set effectively")
#print(df)

#Create a text file output to view the full set
with open('LogTest.txt', 'w') as f:
    f.write(df.to_string(header='Prediction', index=False))
