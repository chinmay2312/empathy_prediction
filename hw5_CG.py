import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

df = pd.read_csv('responses2.csv')

print("Preprocessing")

print("-Filling nulls")
for col in list(df):
    df[col].fillna(df[col].mode()[0], inplace = True)  

print("-Encoding categorical features")
le = LabelEncoder()
categoricalFeatures = []
for col in [id for id,dtyp in df.dtypes.iteritems() if dtyp=='object']:
    categoricalFeatures.append(col)
    df[col] = le.fit_transform(df[col].astype('str'))
df2 = df[categoricalFeatures]
df=df.drop(categoricalFeatures, axis=1)
ohe = OneHotEncoder()
df = df.join(pd.DataFrame(ohe.fit_transform(df2).toarray()))

emp = df['Empathy']
df = df.drop('Empathy',axis=1)

dtree = DecisionTreeClassifier()
knn = KNeighborsClassifier()
mlp = MLPClassifier()

X_train, X_test, y_train, y_test = train_test_split(df, emp, test_size=0.2,random_state = 0)

print("\n Baseline accuracies:")

dtree.fit(X_train, y_train)
print("Decision Tree accuracy: ",dtree.score(X_test, y_test))

knn.fit(X_train, y_train)
print("k-Nearest Neighbors accuracy: ",knn.score(X_test, y_test))

mlp.fit(X_train, y_train)
print("Perceptron accuracy: ",mlp.score(X_test, y_test))


rescaledX = MinMaxScaler(feature_range=(0, 1)).fit_transform(df)
standardizedX = StandardScaler().fit_transform(rescaledX)
normalizedX = Normalizer().fit_transform(standardizedX)

X_train, X_test, y_train, y_test = train_test_split(df, emp, test_size=0.2,random_state = 0)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2,random_state = 0)

acc = 0
for k in range(1,151,10):
	#print("k:",k)
	kbesht = SelectKBest(chi2, k)
	X_new = kbesht.fit_transform(X_train, y_train)
	relevantCols = []
	for col in kbesht.get_support(indices=True):
		relevantCols.append(df.columns[col])
	for nest in range(6,13):
		for dep in range(5,9):
			rfc = RandomForestClassifier(n_estimators=nest,max_depth=dep, random_state=0)
			rfc.fit(X_train[relevantCols], y_train)
			devScore = rfc.score(X_dev[relevantCols], y_dev)
			if devScore > acc:
				acc = devScore
				#print()
				bestk = k
				best_n_est = nest
				best_dep = dep
				#print(len(relevantCols))
				#print("k:",bestk,"\tn_estimators:",best_n_est,"\tdepth:",best_dep)
				#print("RandomForest score on dev:",devScore)

print("\nHyperparameters tuned on basis of dev data for Random Forest:")
print("Feature Count:",bestk,"\tn_estimators:",best_n_est,"\tdepth:",best_dep)
kbesht = SelectKBest(chi2, bestk)
X_new = kbesht.fit_transform(X_train, y_train)
relevantCols = []
for col in kbesht.get_support(indices=True):
	relevantCols.append(df.columns[col])
#print(len(relevantCols))
rfc = RandomForestClassifier(n_estimators=best_n_est,max_depth=best_dep, random_state=0)
rfc.fit(X_train[relevantCols], y_train)
print("RandomForest score on dev: ",rfc.score(X_dev[relevantCols], y_dev))
print("RandomForest score on test:",rfc.score(X_test[relevantCols], y_test))