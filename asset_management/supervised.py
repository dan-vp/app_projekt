import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pyts.classification import KNeighborsClassifier as pyts_KNN
import time
import pickle
from sklearn.linear_model import RidgeClassifierCV
from pyts.transformation import ROCKET

def train_test_split_supervised(data):
    
    # batch_duration wird nicht benötigt, da wir anhand der Dauer pro Schritt trainieren
    data.drop("batch_duration", axis = 1, inplace = True)
    
    # beinhaltet jeden Schritt eines Batches jeweils nur einmal
    only_batch_and_step = data.drop_duplicates(["batchn", "CuStepNo"])

    train_steps = pd.DataFrame(columns = only_batch_and_step.columns)
    test_steps = pd.DataFrame(columns = only_batch_and_step.columns)

    # Führe einen Train-Test-Split pro DevID auf den Trainingsdaten aus
    for devid in only_batch_and_step.DeviationID.unique():
        df_devid = only_batch_and_step[only_batch_and_step.DeviationID == devid]
        train_split, test_split = train_test_split(df_devid, test_size = 0.3, random_state = 42)

        train_steps = pd.concat([train_steps, train_split])
        test_steps = pd.concat([test_steps, test_split])
        
    # Fülle Trainings- und Testdaten mit den anderen Daten, die für only_batch_and_step entfernt wurden

    train_steps = train_steps[["batchn", "CuStepNo"]].merge(data, on = ["batchn", "CuStepNo"])
    test_steps = test_steps[["batchn", "CuStepNo"]].merge(data, on = ["batchn", "CuStepNo"])

    # Laut Anlagenbetreiber ist batchn 255 eine Ausnahme des Normalbetriebs. Daher entfernen wir den Batch aus den Testdaten.
    test_steps = test_steps[test_steps.batchn != 255]
    
    y_train_steps = train_steps.pop("DeviationID")
    y_test_steps = test_steps.pop("DeviationID")

    X_train_steps = train_steps.drop(["batchn", "CuStepNo"], axis = 1)
    X_test_steps = test_steps.drop(["batchn", "CuStepNo"], axis = 1)
    
    return X_train_steps, X_test_steps, y_train_steps, y_test_steps
    

def scale_train_test(X_train, X_test):
    sc_steps = StandardScaler()
    X_train_sc = sc_steps.fit_transform(X_train)
    X_test_sc = sc_steps.transform(X_test)
    
    return X_train_sc, X_test_sc


def create_and_train_knn(X_train, y_train, n_neighbors = 10, weights = "distance", p = 1):
    start = time.time()
    # mit den Werten in weights und p erhöht sich die Accuracy um etwa 2%
    # p = 1 --> Manhattandistanz
    pyts_knn = pyts_KNN(n_neighbors = n_neighbors, weights = weights, p = p)
    pyts_knn.fit(X_train, y_train)
    end = time.time()

    print(f"Trainingszeit auf KNN: {end - start}")
    
    return pyts_knn


def create_and_train_rocket(X_train, y_train, n_kernels, alphas = [0.001], cv = 3):
    rocket = ROCKET(n_kernels= n_kernels)
    start = time.time()
    X_train_rocket = rocket.fit_transform(X_train, y_train)
    end = time.time()

    print(f"Trainingszeit auf ROCKET ({n_kernels} Kernels): {end - start} Sekunden")
    
    rcf = RidgeClassifierCV(alphas = alphas, cv = cv)
    start = time.time()
    rcf.fit(X_train_rocket, y_train)
    end = time.time()
    print(f"Trainingszeit auf RCF: {end - start} Sekunden")
    
    return rocket, rcf


def eval_knn(model, X_test, y_test):
    start = time.time()
    pred = model.predict(X_test)
    end = time.time()

    print(f"Benötigte Zeit zum Predicten: {end - start}")
    eval_df = pd.DataFrame({"pred":pred, "true":y_test})
    
    eval_df["correct"] = eval_df.true == eval_df.pred
    
    return eval_df

def eval_rocket(rocket, rcf, X_test, y_test):
    start = time.time()
    X_test_rocket = rocket.transform(X_test)
    y_pred_rocket = rcf.predict(X_test_rocket)
    end = time.time()

    eval_rocket = pd.DataFrame({"pred": y_pred_rocket, "true":y_test}).reset_index()
    
    print(f"Benötigte Zeit zum Predicten: {end - start}")

    return eval_rocket

def show_eval(eval_df):
    scores = pd.DataFrame.from_dict(classification_report(y_pred = eval_df.pred, y_true = eval_df.true, 
                                                          output_dict = True)).round(4)
    return scores