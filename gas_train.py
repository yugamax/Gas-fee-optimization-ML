import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv(r"dataset\main.eth.csv", on_bad_lines="skip")

df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour

df = df.drop(columns=["name", "hash", "latest_url", "previous_hash", "previous_url", "last_fork_hash", "time"])

df["fee_class"] = pd.cut(df["base_fee"],bins=[-np.inf, 30e9, 60e9, np.inf],labels=[0, 1, 2]).astype(int)

# print(df["fee_class"].value_counts())

newcol = ["high_gas_price", "medium_gas_price", "low_gas_price",
    "high_priority_fee", "medium_priority_fee", "low_priority_fee",
    "base_fee"]

for i in newcol:
    df[i] = df[i] / 1e9

x = df.drop(columns=["base_fee", "fee_class"])
y = df["fee_class"]
# print(df.tail())
over = RandomOverSampler()
x, y = over.fit_resample(x, y)

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.20, random_state=42, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

bestmod = ModelCheckpoint("model\gasfee.keras", monitor='val_accuracy', save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=60, batch_size=16, validation_split=0.20, callbacks=[es, lr, bestmod])

model.save("model\gasfee.keras")
print("Model is saved")

yp = model.predict(x_test)
yp_class = np.argmax(yp, axis=1)
acc = accuracy_score(y_test, yp_class)
print(f"Model accuracy : {(acc*100):.2f} %")

confidences = np.max(yp, axis=1)
print(f"Average confidence: {(np.mean(confidences)*100):.2f} %")

print(classification_report(y_test, yp_class, target_names=["Low", "Mid", "High"]))

# sample = x_test[[0]]
# pred = model.predict(sample)
# pred_class = np.argmax(pred)
# print("Predicted Class:", ["Low", "Mid", "High"][pred_class])

# cm = confusion_matrix(y_test, yp_class)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=["Low", "Mid", "High"],
#             yticklabels=["Low", "Mid", "High"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()
