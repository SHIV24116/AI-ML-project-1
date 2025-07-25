import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
 
# from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score     #to check different models on our dataset and find which one is best 


exercise = input("Enter the name of Exercise you want to perform: ")
file_name= f"{exercise}_posture_dataset.csv"
if not os.path.exists(file_name):
    print(f"{file_name} not found!")
    exit()

df = pd.read_csv(file_name)    
print(df.head())

x = df[["back_angle","elbow_angle","knee_angle"]]
y= df.label

encoder = LabelEncoder()
yle= encoder.fit_transform(y)          # to encode alphabetical data into numerical 0 and 1 as sklearn supports only numerics

df["Encoded_label"]=yle
print(df.head(10))

x_train,x_test,y_train,y_test = train_test_split(x,yle,test_size=0.2)

# cross_val_score(LogisticRegression(solver='liblinear'),x,yle,cv=3)
# cross_val_score(SVC(gamma='auto'),x,yle,cv=3)
# cross_val_score(RandomForestClassifier(n_estimators=40),x,yle,cv=3)
model =RandomForestClassifier(n_estimators=200)
#model = LogisticRegression()
model.fit(x_train,y_train)
#print("Logistic Regression Model trained!")
print("RandomForestClassifier Model trained!")

# y_pred = model.predict(x_test)
# print("\nClassification Report: ")
# print(classification_report(y_test,y_pred))

print("MOdel Accuracy Score: ",model.score(x_test,y_test))

joblib.dump(model,f"{exercise}_model.pkl")
joblib.dump(encoder,f"{exercise}_label_encoder.pkl")
print(f"Model saved as {exercise}_model.pkl")

