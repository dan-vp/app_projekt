import pandas as pd
import pickle
import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras import layers, metrics
import tensorflow as tf
#from keras.layers import Input, Dense
import preprocessing, supervised, unsupervised


def save_model(model_name, model_file_name):

    if model_file_name.split(".")[-1] == "pkl":
        with open(model_file_name, "wb") as file:
            pickle.dump(model_name, file)
    else:
        print("file name has to end with .pkl")

def save_autoencoder(file_name):
    autoencoder.save_weights(f"{file_name}.h5")


df = pd.read_csv("SmA-Four-Tank-Batch-Process_V2.csv", sep = ";")

cleaned_df = preprocessing.pipeline(df)

X_train_steps, X_test_steps, y_train_steps, y_test_steps = supervised.train_test_split_supervised(cleaned_df)
X_train_steps_scaled, X_test_steps_scaled = supervised.scale_train_test(X_train_steps, X_test_steps)

knn = supervised.create_and_train_knn(X_train_steps_scaled, y_train_steps)
eval_knn = supervised.eval_knn(knn, X_test_steps_scaled, y_test_steps)

rocket, rcf = supervised.create_and_train_rocket(X_train_steps_scaled, y_train_steps, 300)
eval_rocket = supervised.eval_rocket(rocket, rcf, X_test_steps_scaled, y_test_steps)

print("Evaluation KNN")
print(supervised.show_eval(eval_knn))
print("Evaluation ROCKET")
print(supervised.show_eval(eval_rocket))

save_model(knn, "asset_management_knn.pkl")
save_model(rocket, "asset_management_ROCKET.pkl")
save_model(rcf, "asset_management_ROCKET_RCF.pkl")




X_train_ae, X_val_ae, X_test_ae, y_train_ae, y_val_ae, y_test_ae, test_ae = unsupervised.train_test_split_autoencoder(cleaned_df)
X_train_ae_tensor, X_val_ae_tensor, X_test_ae_tensor = unsupervised.scale_data_to_tensor_autoencoder(X_train_ae, X_val_ae, X_test_ae)

class AutoEncoder(Model):
        def __init__(self):
            super(AutoEncoder, self).__init__()

            self.encoder = tf.keras.Sequential([
                layers.Dense(15, activation = "relu"),
                layers.Dense(10, activation = "relu"),
                layers.Dense(5, activation = "relu")
            ])

            self.decoder = tf.keras.Sequential([
                layers.Dense(10, activation = "relu"),
                layers.Dense(15, activation = "relu"),
                layers.Dense(X_train_ae.shape[1])


            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


autoencoder = AutoEncoder()
unsupervised.train_autoencoder(autoencoder, X_train_ae_tensor, X_val_ae_tensor)
anomalies = unsupervised.autoencoder_predict(autoencoder, X_train_ae_tensor, X_test_ae_tensor)

unsupervised.show_autoencoder_trained_thresholds(autoencoder, X_train_ae_tensor)

save_autoencoder("asset_management_autoencoder")

eval_ae = unsupervised.eval_autoencoder(anomalies, X_test_ae, y_test_ae)

print("Autoencoder (unsupervised)")
print(unsupervised.show_classifications(eval_ae))
print("0 := no anomaly found, 1 := anomaly found")
print(unsupervised.show_autoencoder_report(eval_ae))

print(unsupervised.show_confusion_matrix(eval_ae))