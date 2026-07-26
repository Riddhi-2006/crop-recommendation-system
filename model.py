import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
import pickle

df= pd.read_csv("Crop_recommendation.csv")

x =df.drop('label',axis =1)
y = df['label']

# Encode crop names into numbers (0 to 21)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=45)

model = RandomForestClassifier(n_estimators=100,random_state=45)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cv_scores = cross_val_score(model,x,y_encoded,cv=5)
print(f"5-Fold CV Mean Accuracy: {cv_scores.mean() * 100:.2f}%")

pickle.dump(model,open('model.pkl','wb'))
pickle.dump(le,open('label_encoder.pkl','wb'))
print("\nModel and LabelEncoder saved successfully as .pkl files!")