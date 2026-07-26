import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
df = pd.read_csv("Crop_recommendation.csv")
print(df.head())
print(df.info())
print(df.columns)
print(df['label'].value_counts())
print(df.isnull().sum())

# Visualisation Crop distirbution
plt.figure(figsize=(12,6))
df['label'].value_counts().plot(kind ='bar')
plt.title("Crop distribution in dataset")
plt.ylabel("counts")
plt.xlabel("Crop label")
plt.tight_layout()
plt.show()

# Visualisation crop relation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.drop('label',axis =1).corr(),annot=True,cmap='coolwarm',fmt =".2f")
plt.title("feature correlation heatmap")
plt.tight_layout()
plt.show()

# visualisation feature boxplots across crops
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for feat in features:
    plt.figure(figsize=(12,4))
    sns.boxplot(x="label",y =feat,data =df)
    plt.xticks(rotation = 90)
    plt.title(f"{feat} requirements by crops")
    plt.tight_layout()
    plt.show()