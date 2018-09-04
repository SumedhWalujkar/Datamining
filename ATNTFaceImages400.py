

def letters_to_digit_convert(letters):
    class_numbers=[]
    for i in range(65,91):
        temp=chr(i) #temp will take values from A to Z i.e ASCI(65)=A, ASCI(66)=B....ASCI(91)=Z
        for j in range(0,len(letters)):# Run the loop with number of loops eequals the length of total letters(So that we can match the letters )
            element=letters[j]    
            if(element==temp): #if the letters match
                char_num=i-65 #if A is found then i will be 65 then A corresponds to 0 when classnumber is i-65
                class_numbers.append(char_num) #append
    
    return class_numbers

###################################################################################################################

def pickData(filename,class_numbers,training_instances,test_instances,choice):
    
    link="C:\\Users\\Sumedh Walujkar\\Documents\\dataminingproject\\Project1\\"+filename
    inputfile=open(link,"r")    
    # readfile
    number_of_classes=10   
    #print(test_instances)
    y=[]
    for line in inputfile:
         y.append(line.split(","))             #get rid of ","
    
    import numpy as np
    arr=np.array(y)
    
    newarr=arr.transpose()                        #finding transpose
    
    img_class=[]    
    noi=len(newarr)
    for x in range(0,noi):
        img_class.append(newarr[x])
    
    for x in range(0,noi):
        img_class[x] = list(map(int,img_class[x])) #converting the string values to int
    
    image_duplicate=img_class 
    
    workable=[]
    workable_reference1=[]
    workable_reference2=class_numbers
    class_numbers=len(workable_reference2)
    
    
    for i in range(0,len(workable_reference2)):
         
         uppervalue=workable_reference2[i]*number_of_classes
         lowervalue=uppervalue+(number_of_classes-1)
         #u=[uppervalue,lowervalue]
         workable_reference1.append(uppervalue)
         workable_reference1.append(lowervalue)
    
    wr=len(workable_reference2)
    for i in range (0,wr):
        uv=2*i
        lv=uv+1
        a=workable_reference1[uv]
        b=workable_reference1[lv]+1 #why +1??
        workable.append(image_duplicate[a:b])
    
    work=[]
    for i in range(0,len(workable)):               #to convert the 3 dimentional list to a two dimentional one
        for j in range(0,number_of_classes):
            work.append(workable[i][j])
    
    #if(choice==1):   
     #   training_instances=10
      #  test_instances=29
    if(choice==0):   
        train_data=[]
        test_data=[]
        
        for x in range(0,class_numbers):
            
            for y in range(0,training_instances):
                temp=y+(x*number_of_classes)
                #temp1=work[temp]
                train_data.append(work[temp])
            for z in range(training_instances,training_instances+test_instances):
                temp2=z+(x*number_of_classes)
                test_data.append(work[temp2])
        
        TrainX=[]
        TrainY=[]
        TestX=[]
        TestY=[]
        attr=[]
        value=[]
        final=[]
    #inpu=int(input("Do you want prediction or k fold validation? \nEnter 0 for prediction and 1 for cross validation\n"))
    
    #if(choice==0):
        for i in range (0,len(train_data)):
            TrainX.append(train_data[i][1:645])
            TrainY.append(train_data[i][0])
        for i in range (0,len(test_data)):
            TestX.append(test_data[i][1:645])
            TestY.append(test_data[i][0])
        
        final.append([0])
        final.append(TrainX)
        final.append(TrainY)
        final.append(TestX)
        final.append(TestY)
        return final
    
    elif(choice==1):
        attr=[]
        value=[]
        final=[]
        for i in range (0,len(work)):
            attr.append(work[i][1:645])
            value.append(work[i][0])
        #for i in range (0,len(test_data)):
            #attr.append(test_data[i][1:321])
            #value.append(test_data[i][0]) 
        
        final.append([1])
        final.append(attr)
        final.append(value)
        
        return final
############################################################################################################
def predictor(final):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn import svm
    from sklearn.cross_validation import cross_val_score
    
    KNN=KNeighborsClassifier()
    cen=NearestCentroid()
    SVM=svm.SVC()
    #if(temper1==0):
    TrainX=[]
    TrainX=final[1]
    TrainY=[]
    TrainY=final[2]
    TestX=[]
    TestX=final[3]
    #predictor(TrainX.TrainY,TestX)
    KNN.fit(TrainX,TrainY)
    cen.fit(TrainX,TrainY)
    SVM.fit(TrainX,TrainY)
    abc=[]
   # print("The predicted values using KNN are", KNN.predict(TestX))
    abc.append(KNN.predict(TestX))
    #print("The predicted values using Centroid are",cen.predict(TestX))
    abc.append(cen.predict(TestX))
    #print("The predicted values using SVM are",SVM.predict(TestX))
    abc.append(SVM.predict(TestX))
    return abc
    ########################################################################################################3
def validator(final):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn import svm
    from sklearn.cross_validation import cross_val_score
    
    KNN=KNeighborsClassifier()
    cen=NearestCentroid()
    SVM=svm.SVC()
    attribute=final[1]
    classlabels=final[2]
    cv=[2,3,5,10]
    average=[]
    k=[]
    c=[]
    s=[]
    #cross_validator(attribute,classlabels,cv)
    for lp in range(0,len(cv)) :
        score=cross_val_score(KNN,attribute,classlabels,cv=cv[lp],scoring="accuracy")
        avg=0
        for ty in range(0,len(score)):
            avg=avg+score[ty]
        avg=avg/len(score)
       
        score1=cross_val_score(cen,attribute,classlabels,cv=cv[lp],scoring="accuracy")
        avg1=0
        for ty in range(0,len(score1)):
            avg1=avg1+score1[ty]
        avg1=avg1/len(score1)
        
        score3=cross_val_score(SVM,attribute,classlabels,cv=cv[lp],scoring="accuracy")
        avg3=0
        for ty in range(0,len(score3)):
            avg3=avg3+score3[ty]
        avg3=avg3/len(score3)
        demoavg=[]
        demoavg.append(avg)
        demoavg.append(avg1)
        demoavg.append(avg3)
        
        average.append(demoavg)
    for i in range(0,len(average)):
        k.append(average[i][0])
        c.append(average[i][1])
        s.append(average[i][2])
    import matplotlib.pyplot as plt
    plt.plot(k)
    plt.plot(c)
    plt.plot(s)
    plt.text(0.0, 0.975, r'Blue=Knn  Red=Centroid Green=SVM')
    plt.xlabel("0=cv(2) 1=cv(3) 2=cv(5) 3=cv(10)")
    plt.show()
    
#########################################################################################################


def dodande(filename,class_numbers,noctc):
    from sklearn.metrics import accuracy_score
    runsequences=int(input("give the number of run sequences"))
    ender=[]
    
    #class_numbers=letters_to_digit_convert(letters)
    for i in range(0,runsequences):
        
        abc=[]
        training_instances=int(input("Enter the number of training elements for instance " ))
        test_instances=noctc-training_instances
        print("The test Instances are:\n")
        print(test_instances)
        final=pickData(filename,class_numbers,training_instances,test_instances,0)
        
        temporary=predictor(final)
        
        for t in range(0,len(temporary)):
            abc.append(accuracy_score(temporary[t], final[4]))
        ender.append(abc)
    k=[]
    c=[]
    s=[]
    for i in range(0,len(ender)):
        k.append(ender[i][0])
        c.append(ender[i][1])
        s.append(ender[i][2])
    import matplotlib.pyplot as plt
    plt.plot(k)
    plt.plot(c)
    plt.plot(s)
    plt.text(0.0, 0.975, r'Blue=Knn  Red=Centroid Green=SVM')
    #plt.xlabel("0=cv(2) 1=cv(3) 2=cv(5) 3=cv(10)")
    plt.show()    
    return ender;


#########################################################################################################
    
print("Place the file you want to perform analysis on in the C drive")

filename=input("Enter the filename:\n")
filename=filename+".txt"
noctbc=int(input("Enter the number of FACEIMAGES you want to consider:\n"))
number_of_classes=int(input("Input the number of classes of each FACEIMAGE:\n"))
class_numbers=[]
print("Enter the FACEIMAGE number:")
for i in range(0,noctbc):
    class_numbers.append(int(input()))

    
cghi=int(input("do you want to perform D and E bit?\n enter 0 for yes and 1 for no:\n"))
ender=[]
if(cghi==0):
    
    ender=dodande(filename,class_numbers,number_of_classes)
    
    
choice=int(input("Do you want to perform prediction or cross validation\nEnter 0 for prediction and 1 for cross validation:\n"))

if(choice==0):
    
    training_instances=int(input("Enter the number of training elements:\n"))
    
    test_instances=number_of_classes-training_instances
elif(choice==1):
    
    training_instances=0
    test_instances=0



#class_numbers=letters_to_digit_convert(letters)



final=pickData(filename,class_numbers,training_instances,test_instances,choice)
temper1=final[0][0]
temporary=[]
if(temper1==0):

    temporary=predictor(final)
    from sklearn.metrics import accuracy_score
    print("\n\n\nThe Accuracy Scores are:")
    for t in range(0,len(temporary)):
        print(accuracy_score(temporary[t], final[4]))
    print("for KNN,Centroid and SVM respectively")
    
elif(temper1==1):

    validator(final)

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10  23:54:56 2017

@author: Sumedh Walujkar 


"""

    



    
