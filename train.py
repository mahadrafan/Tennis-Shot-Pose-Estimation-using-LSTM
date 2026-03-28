import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

CSV_PATH    = "tennis_dataset.csv"
MODEL_PATH  = "tennis_model.h5"
LABELS_PATH = "tennis_labels.json"
SEQ_LENGTH  = 30    
N_FEATURES  = 99    

def build_sequences(df, seq_length=30):
    X, y = [], []
    labels = df["label"].values
    features = df.drop("label", axis=1).values.astype(np.float32)

    i = 0
    while i + seq_length <= len(df):
        window_labels = labels[i:i+seq_length]
        if len(set(window_labels)) == 1:
            X.append(features[i:i+seq_length])
            y.append(window_labels[0])
            i += seq_length 
        else:
            i += 1  

    return np.array(X), np.array(y)

def build_model(n_classes, seq_length=30, n_features=99):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_training(history, output_path="training_plot.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    print(f"training plot saved: {output_path}")

def save_confusion_matrix(cm, classes, path="confusion_matrix.png"):

    plt.figure(figsize=(6,5))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):

            plt.text(j, i, cm[i,j],
                     ha="center",
                     va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(path, dpi=120)

    print("Saved:", path)

def save_classification_report(report_df, path="classification_report.png"):
    report_df = report_df.round(3)

    fig, ax = plt.subplots(figsize=(10,4))

    ax.axis("off")

    table = ax.table(
        cellText=report_df.values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.title("Classification Report", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(path, dpi=200)

    print("Saved:", path)

def main():
    print("="*60)
    print("LSTM TRAINING — Tennis Shot Classifier")
    print("="*60)

    # Load data
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"\nData loaded: {len(df)} frame")
    print("Label Distribution:")
    print(df["label"].value_counts().to_string())

    required = {"forehand", "backhand", "serve"}
    available = set(df["label"].unique())
    missing = required - available
    if missing:
        print(f"\nWarning: missing class in the dataset: {missing}")
        print("continue training with the remaining class? (y/n): ", end="")
        if input().strip().lower() != "y":
            return

    print(f"\building sequens (window={SEQ_LENGTH} frame)...")
    X, y_raw = build_sequences(df, SEQ_LENGTH)
    print(f"Total sekuens: {len(X)}  shape: {X.shape}")

    if len(X) < 50:
        print("Error: sekuens too little. add more video.")
        return

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    y_cat = to_categorical(y_encoded)
    n_classes = len(le.classes_)
    print(f"class: {list(le.classes_)}")

    label_map = {i: cls for i, cls in enumerate(le.classes_)}
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)
    print(f"Label mapping saved: {LABELS_PATH}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}")

    model = build_model(n_classes, SEQ_LENGTH, N_FEATURES)
    model.summary()

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5, verbose=1),
    ]

    print("\start training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "="*60)
    print("evaluation on the validation set")
    print("="*60)
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

    model.save(MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    plot_training(history)

    val_acc = max(history.history["val_accuracy"])
    print(f"\nBest val accuracy: {val_acc*100:.1f}%")

if __name__ == "__main__":
    main()
