import pandas as pd #here we import pandas library as pd using Alice method
content=pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")#here we read the csv file using pd(pandas)
display(content)#here we used display function for printing in good formating like as tables
#print(content) we have an option of print but that cannot print in the proper shape

#In this snippet of code we do initial inspection
content.head()#here content means our data and .head() funcation means gives only first five rows

#according to our recruitments we print/display first 10 rows
content.head(10) # In the head() funcation we give an argument that decided how many rows you want

content.info()
#using info() we get
#total rows count,index range,names of colums,data type of every colum, Non null values count,memory usag

content.describe()# describe() funcation is used to see the quick statistics summery of numerical data

#In this portion we clean the missing values
display(content["Age"])#for seeing the NaN values
mean = content["Age"].mean()#in this line we get mean from all ages
content["Age"].fillna(mean,inplace=True) # we used fillna() for filling null values with mean
#note we use inplace=True that's means changed in the original data
#note if we use inplace=False that's means return another othe object without changing original
display(content["Age"]) # for work clearfication
most=content["Embarked"].mode()[0] #mode funcation return the value that most repeated
content["Embarked"].fillna(most,inplace=True)
display(content) # for clearfication
content.drop("Cabin",axis=1,inplace=True)

#Feature Engineering (Creating New Columns)

# Step 1: FamilySize
content["FamilySize"] = content["SibSp"] + content["Parch"]

# Step 2: IsAlone
content["IsAlone"] = (content["FamilySize"] == 0).astype(int)

display(content)

# survival rate by passengerclass
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.barplot(x="Pclass", y="Survived", data=content,)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

#Age v fare color by survived
plt.figure(figsize=(8,6))
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=content, palette={0:"red",1:"yellow"})
plt.title("Age vs Fare (colored by Survival)")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.gca().set_facecolor("white")   # apni pasand ka color

plt.show()

#Now we dropping the other values that create the confusion during the modeling
content.drop(["Name", "Ticket", "Sex", "Embarked"], axis=1, inplace=True)
