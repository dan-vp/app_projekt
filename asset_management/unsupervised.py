# Trainiere einen Autoencoder auf skalierten Daten
# Über eine Encoderschicht mit ReLu-Aktivierungsfunktion und mit 2 Neuronen sollen die Daten rekonstruiert werden
# Weichen die rekonstruierten Daten zu stark von den originalen Inputdaten ab, meldet der Autoencoder eine Anomalie

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras import layers, metrics

        
def train_autoencoder(autoencoder, X_train_tensor, X_val_tensor, optimizer =  SGD(learning_rate=0.005, momentum = 0.5), loss = "mse"):

    autoencoder.compile(optimizer = optimizer, loss = "mse")


    autoencoder.fit(X_train_tensor, X_train_tensor,
                    epochs=10,
                    batch_size=32,
                    shuffle=True,
                   validation_data = (X_val_tensor, X_val_tensor)
                   )


def train_test_split_autoencoder(data):
    
    # 60:20:20 Train-Validation-Test-Split der Daten im Normalzustand

    normal_batches = data[data.DeviationID == 1.0].batchn.nunique()
    normal_to_train = int(normal_batches * 0.6)
    normal_to_val = int(normal_batches * 0.8)

    normal_df = data[data.DeviationID == 1.0]
    
    normal_df = normal_df[normal_df.batchn != 255] 
    # laut Anlagenbetreiber ist Batch Nr. 255 eine Ausnahme und kann daher entfernt werden
    
    # alle normalen Batches bis zu dieser Nummer werden im Trainingsdatensatz benutzt
    train_batchn_max = normal_df.batchn.unique()[normal_to_train]
    # alle normalen Batches bis zu dieser Nummer werden im Validierungsdatensatz benutzt
    val_batchn_max = normal_df.batchn.unique()[normal_to_val]
    # Die restlichen normalen Batches werden mit fehlerhaften Batches im Testdatensatz getestet
    
    train_ae = normal_df[normal_df.batchn <= train_batchn_max]
    val_ae = normal_df[(normal_df.batchn > train_batchn_max) & (normal_df.batchn <= val_batchn_max)]
    test_ae = normal_df[normal_df.batchn > val_batchn_max]
    # Füge die fehlerhaften Batches hinzu und sortiere den Testdatensatz chronologisch
    test_ae = pd.concat([test_ae, data[data.DeviationID > 1]]).sort_values(["batchn", "CuStepNo", "step_length"])
    
    y_train_ae = train_ae.pop("DeviationID")
    y_test_ae = test_ae.pop("DeviationID")
    y_val_ae = val_ae.pop("DeviationID")

    X_train_ae = train_ae.drop(["batchn", "CuStepNo", "LIC23002_SP", "LevelMainTank"], axis = 1)
    X_val_ae = val_ae.drop(["batchn", "CuStepNo", "LIC23002_SP", "LevelMainTank"], axis = 1)
    X_test_ae = test_ae.drop(["batchn", "CuStepNo", "LIC23002_SP", "LevelMainTank"], axis = 1)

    return X_train_ae, X_val_ae, X_test_ae, y_train_ae, y_val_ae, y_test_ae, test_ae
    

def scale_data_to_tensor_autoencoder(X_train, X_val, X_test):
    
    sc_ae = StandardScaler()

    X_train_ae = sc_ae.fit_transform(X_train)
    X_val_ae = sc_ae.fit_transform(X_val)
    X_test_ae = sc_ae.transform(X_test)
    
    X_train_tensor =  np.asarray(X_train_ae).astype('float32')
    X_test_tensor =  np.asarray(X_test_ae).astype('float32')
    X_val_tensor =  np.asarray(X_val_ae).astype('float32')
    
    return X_train_tensor, X_val_tensor, X_test_tensor


def autoencoder_predict(autoencoder, X_train_tensor, X_test_tensor):
    # Definiere einen Schwellenwert für jedes Feature (threshold) und speichere Überschreitungen dieser
    # Schwellenwerte als 1 in einer Matrix mit der selben Form wie die Testdatenmatrix
    
    # Anomalie wird mit 1 markiert, sobald der gemessene skalierte Wert pro Datenpunkt größer ist als:
    # durchschnittliche Differenz zwischen Vorhersage eines Features und tatsächlicher Wert des Features + 2x Standardabweichung der Differenz

    # Bsp.: Anomalie bei 5. Datenpunkt in Feature Nr. 2, aber sonst in keinem anderen Feature
    # --> Zeile 5: 0 1 0 0 0 0 0 0 ... 0 0 0
    threshold = np.mean(autoencoder.predict(X_train_tensor) - X_train_tensor, axis = 0) + 2 * np.std(autoencoder.predict(X_train_tensor) - X_train_tensor, axis = 0)
    anomalies = np.where(autoencoder.predict(X_test_tensor) - X_test_tensor > threshold, 1, 0)
    
    sums = np.sum(anomalies, axis = 1)
    
    print(f"Im Testdatensatz mit {len(sums)} Datenpunkten werden an {len(sums[sums > 0])} Datenpunkten Anomalien gemeldet.")
    
    return anomalies

def show_autoencoder_trained_thresholds(autoencoder, X_train_tensor):
    return np.mean(autoencoder.predict(X_train_tensor) - X_train_tensor, axis = 0) + 2 * np.std(autoencoder.predict(X_train_tensor) - X_train_tensor, axis = 0)


def eval_autoencoder(anomalies, X_test, y_test_ae):
    eval_ae = pd.DataFrame(anomalies, columns = X_test.columns)

    y_test_ae = y_test_ae.reset_index(drop = True) # für Merge in der nächsten Zeile

    eval_ae["y_true"] = y_test_ae
    eval_ae["pred_anomalies"] = np.sum(anomalies, axis = 1)
    eval_ae["anomalies_found"] = eval_ae["pred_anomalies"] > 0

    eval_ae["y_real_binary"] = eval_ae.y_true - 1
    eval_ae.loc[eval_ae.y_true > 1, "y_real_binary"] = 1
    
    return eval_ae


def show_classifications(eval_ae):
    return eval_ae.groupby(["y_true", "anomalies_found"]).agg(anomalies_found = ("anomalies_found", "count"))

def show_autoencoder_report(eval_ae):
    autoencoder_report = classification_report(y_true = eval_ae.y_real_binary, y_pred = eval_ae.anomalies_found, output_dict = True)
    autoencoder_report = pd.DataFrame.from_dict(autoencoder_report).round(4)
    
    return autoencoder_report

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_confusion_matrix(eval_ae):
    return ConfusionMatrixDisplay(confusion_matrix(eval_ae.y_real_binary, eval_ae.anomalies_found, normalize = "true")).plot()
