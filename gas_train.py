import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint

df = pd.read_csv(r"main.eth.csv", on_bad_lines="skip")

df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour

df = df.drop(columns=["name", "hash", "latest_url", "previous_hash", "previous_url", "last_fork_hash", "time"])

df["fee_class"] = pd.qcut(df["base_fee"], q=3, labels=[0, 1, 2]).astype(int)

print(df.head())

X = df.drop(columns=["base_fee", "fee_class"])
y = df["fee_class"]

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

bestmod = ModelCheckpoint("skin_test.keras", monitor='val_accuracy', save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=90, batch_size=16, validation_split=0.1, callbacks=[es , lr , bestmod])
model.save("gasfee.keras")
print("Model is saved")
loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

sample = x_test[[0]]
pred = model.predict(sample)
pred_class = np.argmax(pred)
print("Predicted Class:", ["Low", "Mid", "High"][pred_class])
