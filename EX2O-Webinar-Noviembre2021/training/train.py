import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# cargamos datos
data = load_iris()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.3, random_state=4)


# entrenamiento del modelo
model = LogisticRegression(C=0.1, 
                           max_iter=20, 
                           fit_intercept=True, 
                           solver='liblinear')
model.fit(Xtrain, Ytrain)



pkl_filename = "modelo.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)