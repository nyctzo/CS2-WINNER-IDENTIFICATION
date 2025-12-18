from sklearn.model_selection import train_test_split
X, y = df_selected.drop(["t_win"], axis= 1 ), df_selected["t_win"]
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

scaler = StandardScaler()

X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#kneighboursclassifier

knn = KNeighborsClassifier()
knn.fit(X_trained_scaled, y_train)

from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    "n_neighbors": list(range(5,17,12)),
    "weights": ["uniform" , "distance"]
}


knn = KNeighborsClassifier(n_jobs= 4)
clf = RandomizedSearchCV(knn, param_grid, n_jobs=4, n_iter = 3, verbose = 2 , cv=3)
clf.fit(X_trained_scaled, y_train)

#randomforestclassifier

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_jobs = 4)
forest.fit(X_trained_scaled, y_train)

