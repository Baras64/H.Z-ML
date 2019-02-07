
from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
Rcancer = pd.read_csv('C:\\Users\Zubin\\Desktop\\Tsec Hackathon\\Data.txt', sep=",", header=None,
                     names=["Sample code number", "Clump Thickness",
                            "Uniformity of Cell Size", "Uniformity of Cell Shape",
                            "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei" ,
                            "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"])#importing dataset

#deleting missing values
Rcancer=Rcancer.replace(to_replace="?", value=np.nan)
cancer=Rcancer.dropna()

 #Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, f1_score, classification_report, confusion_matrix

#reshaping the dataset
X=pd.DataFrame(cancer,columns=["Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
                            "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei" ,
                            "Bland Chromatin", "Normal Nucleoli","Mitoses"])

y=pd.DataFrame(cancer, columns=["Class"])

    #Split Validation
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.4, random_state=1, stratify=y)

#building scikit-learn model
loR = LogisticRegression(C=1e5, random_state=1)#instantiate model
loR.fit(X_train, y_train)#fitting dataframe into the model
y_pred = loR.predict(X_test)

#model evaluation

 #2 for benign and 4 for malignant
logit_matrix = confusion_matrix(y_test, y_pred)

    #Single Regression Prediction

from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs

    #generate 2d calssification dataset
X,y = make_blobs(n_samples=100, centers=2, n_features=2,random_state=1)
    #fit final model
    #defining datasets
Xnew = [[0,0,0,0,0,0,0,0,0]]

def input2():
    Age = int(input("Enter your Age:  "))
    Gender = input("Enter your Gender:  ")
    print("Enter values between 0 to 10")
    print("Strongly Disagree \t Neutral \t Strongly Agree")
    print("0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10")

    Xnew[0][0]=int(input("I am a frequent Smoker: "))
    Xnew[0][1] = int(input("I drink a lot of alcohol: "))
    Xnew[0][2] = int(input("I frequently use sunscreen "))
    Xnew[0][3] = int(input("A lot of people have cancer in my family: "))
    Xnew[0][4] = int(input("I regularly eat beetle quid "))
    Xnew[0][5] = int(input("I am always outside in the polluted regions "))
    Xnew[0][6] = int(input("I am out in the Sun a lot "))
    Xnew[0][7] = int(input("I eat a lot of Genetically modified food: "))
    Xnew[0][8] = int(input("I always eat food cooked in Hydrogenated oil "))
    sum = 0
    for i in range (0,9):
         sum= sum+ Xnew[0][i]
    print("Percentage of having Cancer=",(sum/90*100))
    value=Prediction()
    if value == 2:
        print("Tumour Type predicted as benign or completely absent")
    else:
        print("Tumour Type predicted as Malignant, please consult a doctor")
    Insurances()
    Treatment()

def Prediction():
    #Xnew = [[8,10,10,8,7,10,9,7,1]]
    ynew = loR.predict(Xnew)
    Predicted_Value = ynew[0]
    for i in range(len(Xnew)):
        print("X=%s, Predicted=%s"% (Xnew[i], ynew[i]))#printing input and the predicted data
    return Predicted_Value

def Insurances():
    Rcancer2 = pd.read_csv('C:\\Users\Zubin\\Desktop\\Tsec Hackathon\\Data.txt', sep=",", header=None,
                           names=["Fraction_Genome_Altered", "Histological_Type",
                                  "Sex"])

    # deleting missing values
    Rcancer2 = Rcancer2.replace(to_replace="?", value=np.nan)
    cancer2 = Rcancer2.dropna()

    # Regression Model
    X2 = pd.DataFrame(cancer2, columns=["Histological_Type", "Sex"])
    y2 = pd.DataFrame(cancer2, columns=["Fraction_Genome_Altered"])

    # Split Validation
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.4, random_state=1, stratify=y2)

    loR2 = LogisticRegression(C=1, random_state=1)
    loR2.fit(X_train2, y_train2)
    y_pred2 = loR2.predict(X_test2)

    # Single Regression Prediction
    # generate 2d classification dataset
    X2, y2 = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

    # defining datasets
    Xnew2 = [[0, 0]]

    def input4():
        print("\n\n\n ###############INSURANCE POLICIES####################")
        print("1.Male 2.Female")
        Xnew2[0][1] = int(input("Enter your Gender:  "))
        print("Enter your choice of Histological Code")
        print("1.Glioma\n2.Osteosarcoma\n3.Carcinoma\n4.Haematopoietic Neoplasm\n5.Malignant Melanoma\n6.Rhabdomyosarcoma\n7.Ewings Sarcoma-Peripheral Primitive Neuroectodermal Tumour\n8.Esothelioma\n9.Lymphoid Neoplasm\n10.Chondrosarcoma\n11.Neuroblastoma\n12.Primitive Neuroectodermal Tumour-Medulloblastoma\n13.Other\n14.Giant Cell Tumour\n15.Fibrosarcoma\n16.Carcinoid-Endocrine Tumour")
        Xnew2[0][0] = int(input("Histological Code "))+2

        value = Prediction()
        if Xnew2[0][1] == 1:
            if value[0] >= 0.0 and value[0] <= 0.4:
                print("Mild Condition ")
                print("Standard policy\nSum amount 2-25lakhs and no additional premiums ")
            elif value[0] > 0.4 and value[0] <= 0.6:
                print("Intense Condition")
                print("Exclusive policy\nSum amount 2-35lakhs and no additional premiums")
            else:
                print("Extreme Condition")
                print("Emergency Policy\nSum amount 35-200lakhs and 14k rs additional premium")
        else:
            if value[0] >= 0.0 and value[0] <= 0.4:
                print("Mild Condition ")
                print("Standard policy\nSum amount 2-25lakhs and 9k rs premium ")
            elif value[0] > 0.4 and value[0] <= 0.6:
                print("Intense Condition")
                print("Exclusive policy\nSum amount 2-35lakhs and 14k rs additional premiums")
            else:
                print("Extreme Condition")
                print("Emergency Policy\nSum amount 35-200lakhs and 24k rs additional premium")

    def Prediction():
        # Xnew = [[8,10,10,8,7,10,9,7,1]]
        ynew2 = loR2.predict_proba(Xnew2)
        Predicted_Value = ynew2[0]
        print(Predicted_Value[0])
        return Predicted_Value

    input4()

def Treatment():
    Rcancer1 = pd.read_csv('C:\\Users\Zubin\\Desktop\\Tsec Hackathon\\Data.txt', sep=",", header=None,
                           names=["Patient ID", "Breast Cancer", "Prostate Cancer", "Basal Cell Cancer", "Melanoma",
                                  "Colon Cancer", "Lung Cancer", "Cervical Cancer", "Lymphoma", "Treatment",
                                  "Tumour Type"])

    # deleting missing values
    Rcancer1 = Rcancer1.replace(to_replace="?", value=np.nan)
    cancer1 = Rcancer1.dropna()

    # Regression Model
    X1 = pd.DataFrame(cancer1, columns=["Breast Cancer", "Tumour Type"])
    y1 = pd.DataFrame(cancer1, columns=["Treatment"])

    # Split Validation
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.4, random_state=1, stratify=y1)

    loR1 = LogisticRegression(C=1e5, random_state=1)
    loR1.fit(X_train1, y_train1)
    y_pred1 = loR1.predict(X_test1)

    # Single Regression Prediction
    # generate 2d calssification dataset
    X1, y1 = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
    # defining datasets

    Xnew1 = [[0, 0]]

    def input3():
        print("\n\n\n#################TREATMENT###################")
        print("Enter data for appropriate Treatment")
        print("Enter your choice")
        print("1.Breast Cancer\n2.Prostate Cancer\n3.Basal Cell Cancer\n4.Melanoma\n5.Colon CancerLung Cancer\n6.Cervical Cancer\n7.Lymphoma")

        Xnew1[0][0] = int(input("Type of Cancer: "))

        print("2.Benign 4.Malignant")
        Xnew1[0][1] = int(input("Type of Tumour: "))
        f = Xnew1[0][0]
        se = Xnew1[0][1]
        value = Prediction()
        if f != 0 or se != 0:
            if value == 1:
                print("Best Treatment would be Chemotherapy ")
            elif value == 2:
                print("Best Treatment would be Radiation Therapy")
            elif value == 3:
                print("Best Treatment would be Surgery")
            elif value == 4:
                print("Best Treatment would be Immunotherapy")
            elif value == 5:
                print("Best Treatment would be Hormone Therapy")
            elif value == 6:
                print("Best Treatment would be Stem Cell Transplant")
            elif value == 7:
                print("Best Treatment would be Targeted Therapy")
            elif value == 8:
                print("Best Treatment would be Precision Medicine ")
            elif value == 9:
                print("Blood Transfusion")
            else:
                print("No Treatment ")
        else:
            print("No Treament Needed")

    def Prediction():
        ynew1 = loR1.predict(Xnew1)
        Predicted_Value1 = ynew1[0]
        for i in range(len(Xnew1)):
            print("X=%s, Predicted=%s" % (Xnew1[i], ynew1[i]))
        return Predicted_Value1

    input3()


input2()
