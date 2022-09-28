import random
import pandas as pd

#example data set taken from the iris.data file
TinyData = [
    [5.1,3.5,1.4,0.2,"Iris-setosa"],
    [4.9,3.0,1.4,0.2,"Iris-setosa"],
    [4.7,3.2,1.3,0.2,"Iris-setosa"],
    [7.0,3.2,4.7,1.4,"Iris-versicolor"],
    [6.4,3.2,4.5,1.5,"Iris-versicolor"],
    [6.9,3.1,4.9,1.5,"Iris-versicolor"],
    [6.3,3.3,6.0,2.5,"Iris-virginica"],
    [5.8,2.7,5.1,1.9,"Iris-virginica"],
    [7.1,3.0,5.9,2.1,"Iris-virginica"],
]

#Test points are in the same format as the Training Data, the key difference being that
#for Training Data, we already know which type of Iris this is, but with Test Points we are using
#a prediction in the final (class) slot
TestPoint = [[5.6,4.1,1.2,0.5,"Iris-setosa"]]

def Euclidean(p1, p2):
    #dist is the distance between the test point (p1) and one of the training points (p2)
    subtracted = list()
    dist = 0

    #loop through each dimension of the data and perform Euclidean Distance calculation
    #while ignoring the last spot which contains the class of the datum, then append those
    #values to a list for summation and squaring
    for i in range(len(p1)-2):
        subtracted.append(p1[i] - p2[i])

    #loop through each dimension in the subtracted list and square them
    for i in range(len(subtracted)):
        subtracted[i] = (subtracted[i] ** 2)
    
    #take the square root of the sum of the subtracted values to get the Euclidean distance
    #between the two points
    dist = sum(subtracted) ** (1/2)

    return dist

#Perform KNN with Euclidean Distance
def kNN(k, training, testpoint):
    
    #Variables to collect the evaluated predictions
    sxs = 0
    sxve = 0
    sxvi = 0
    vexs = 0
    vexve = 0
    vexvi = 0
    vixs = 0
    vixve = 0
    vixvi = 0

    #Select the number of nearest neighbors based on the provided k value
    for i in range(len(testpoint)):
        dists = list()

        #print(str(k) + " Nearest Neighbors to " + str(testpoint[i]))
        
        for l in range(len(training)):
            dists.append(Euclidean(training[l], testpoint[i]))
        
        #Sort both the training points and the dists such that they preserve their combinations
        #then insert the values into a dictionary so that they are ordered from least to greatest
        #distances, makes it easy to select them for kNN
        for t in range(0, len(training)):
            for z in range(t+1, len(training)-1):
                if ((dists[z] <= dists[t])):
                    temp = dists[t]
                    dists[t] = dists[z]
                    dists[z] = temp

                    temp2 = training[t]
                    training[t] = training[z]
                    training[z] = temp2
        
        setosa = 0
        versicolor = 0
        virginica = 0
        for v in range(0,k):
            train = training[v]
            
            if train[4] == "Iris-setosa":
                setosa += 1
                #print("setosa " + str(setosa))
            elif train[4] == "Iris-versicolor":
                versicolor += 1
                #print("versicolor " + str(versicolor))
            elif train[4] == "Iris-virginica":
                virginica += 1
                #print("virginica " + str(virginica))
        test = testpoint[i]

        #choose our prediction based off the greatest number of neighbors reported per each kind
        #and then add 1 to the matrix variable which corresponds with the guess
        if (setosa > versicolor) & (setosa > virginica):
            if test[4] == "Iris-setosa":
                sxs += 1
            elif test[4] == "Iris-versicolor":
                sxve += 1
            elif test[4] == "Iris-virginica":
                sxvi += 1

        elif (versicolor > setosa) & (versicolor > virginica):
            if test[4] == "Iris-setosa":
                vexs += 1
            elif test[4] == "Iris-versicolor":
                vexve += 1
            elif test[4] == "Iris-virginica":
                vexvi += 1

        elif (virginica > setosa) & (virginica > versicolor):
            if test[4] == "Iris-setosa":
                vixs += 1
            elif test[4] == "Iris-versicolor":
                vixve += 1
            elif test[4] == "Iris-virginica":
                vixvi += 1
        
    matrix = [
    [sxs, sxve, sxvi],
    [vexs, vexve, vexvi],
    [vixs, vixve, vixvi]]
    
    Accuracy = ConfusionMatrix(matrix, sxs, sxve, sxvi, vexs, vexve, vexvi, vixs, vixve, vixvi)

    return Accuracy

#Take the true/false positive/negative prediction values and
#calculate Accuracy, Precision, Recall, and F1 score
def ConfusionMatrix(matrix, sxs, sxve, sxvi, vexs, vexve, vexvi, vixs, vixve, vixvi):
    print("Confusion Matrix")
    print(matrix)
    print()

    #Calculate precision for each class
    SetPrec =  (sxs) / (sxs + sxve + sxvi)
    VerPrec = (vexve) / (vexs + vexve + vexvi)
    VirPrec = (vixvi) / (vixs + vixve + vixvi)

    #Calculate recall for each class
    SetRec = (sxs) / (sxs + vexs + vixs)
    VerRec = (vexve) / (vexve + sxve + vixve)
    VirRec = (vixvi) / (vixvi + vexvi + sxvi)

    #Calculate F1-Score for each class
    SetF1 = 2 * ((SetPrec * SetRec) / (SetPrec + SetRec))
    VerF1 = 2 * ((VerPrec * VerRec) / (VerPrec + VerRec))
    VirF1 = 2 * ((VirPrec * VirRec) / (VirPrec + VirRec))

    Accuracy = (sxs + vexve + vixvi) / (sxs + sxve + sxvi + vexs + vexve + vexvi + vixs + vixve + vixvi)

    #Calculate Macro and Weighted Averages, since we don't actually have supports here though, they'll all be equally weighted
    PrecMacroAvg = (SetPrec + VerPrec + VirPrec) / 3
    RecMacroAvg = (SetRec + VerRec + VirRec) / 3
    F1MacroAvg = (SetF1 + VerF1 + VirF1) / 3


    d = {"": ["Setosa", "Versicolor", "Virginica", "", "Accuracy", "Macro Avg"], 
         "Precision": [SetPrec, VerPrec, VirPrec, "", "", PrecMacroAvg,],
         "Recall": [SetRec, VerRec, VirRec, "", "", RecMacroAvg],
         "F1-Score": [SetF1, VerF1, VirF1, "", Accuracy, F1MacroAvg]}
    
    df = pd.DataFrame(d)
    print(df)
    
    return Accuracy

#Read data from the iris.data file into a list of lists
def ReadIn():
    data = list()

    with open("iris.data") as file:
        for i in range(0,150):
            #read the file in line by line as a single string
            line = file.readline().strip()
            
            #split the string into a list of values along the comma separations
            lineList = line.split(",")
            
            #iterate through the list of strings and change the numbers to floats for kNN
            for j in range(len(lineList)-1):
                lineList[j] = float(lineList[j])

            #add the lists to a master list to use for kNN
            data.append(lineList)

            #iterate through the number of lines of data in the file we are using
            i+=1
    return data

def BestK(val, train, test):
    ks = [1,2,3,4,5,6,7,8,9,10]
    best = 0
    thisRun = 0
    bestK = 0

    for each in ks:
        thisRun = kNN(each, val, test)
        if (thisRun > best):
            best = thisRun
            bestK = each

    print()
    print(str(bestK) + " gave us the highest accuracy.")
    print("So, lets run our training and testing data using that K.")
    print()
    Showcase(bestK, train, test)
    return

def Showcase(bestK, set1, set2):
    kNN(bestK, set1, set2)
    return

#Create the MasterSet and then 3 empty lists that will get populated with random data from
#the MasterSet so that we have an evenly random distribution to test our ML Alg on
MasterSet = ReadIn()
Training = list()
Validation = list()
Testing = list()

#shuffle the MasterSet so that we can have a random distribution across our three
#data sets to ensure that our testing and training is adequate
random.shuffle(MasterSet)

#split randomized MasterSet into three different data sets to use for Training, Validation, and
#testing
for i in range(0,49):
    Training.append(MasterSet[i])
for j in range(50,99):
    Validation.append(MasterSet[j])
for k in range(100,149):
    Testing.append(MasterSet[k])


#Run BestK to finish it out
BestK(Validation, Training, Testing)






