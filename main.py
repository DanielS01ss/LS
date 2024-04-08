from fastapi import FastAPI
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pyod.models.knn import KNN

app = FastAPI()

neigh = None 
clf = NotImplemented

@app.on_event("startup")
def load_train_model():
    df = pd.read_csv("./iris_cleaned.csv")
    global neigh
    neigh = KNeighborsClassifier(n_neighbors=len(np.unique(df['Y'])))
    neigh.fit(df[df.columns[:4]].values.tolist(),df['Y'])
    global clf
    clf = KNN()
    clf.fit(df[df.columns[:4]].values.tolist(),df['Y'])
    print("Training done!")

@app.get("/anomaly")
def anomaly(p1: float, p2: float, p3:float, p4: float):
    pred = clf.predict([p1,p2,p3,p4])
    return "{}".format(pred[0])

@app.get("/predict")
def predict(p1:float, p2:float, p3:float, p4:float):
    pred = neigh.predict([[p1,p2,p3,p4]])
    return "{}".format(pred[0])

@app.get("/")
def read_root():
    return {"Hello":"World"}