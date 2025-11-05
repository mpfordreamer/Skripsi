import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# data folder
dataset_dir = os.path.join(os.path.dirname(__file__), "CS_Train")
mat_files = sorted(glob.glob(os.path.join(dataset_dir, "*.mat")))
print("Found MAT files:", [os.path.basename(f) for f in mat_files])

overall = []  # to collect a brief summary across participants

for file_path in mat_files:
    input_file = os.path.basename(file_path)
    print("\n" + "="*80)
    print(f"====================== PARTICIPANT: {input_file} ======================")
    print("="*80)

    # Load one participant file
    data = sio.loadmat(file_path)

    # Load variables
    X = data["data"]  # Shape: (60, 64)
    y_v = data["valence_labels"].flatten()
    y_a = data["arousal_labels"].flatten()
    y_d = data["dominance_labels"].flatten()

    print("X.shape:", X.shape)

    # Reshape X to (12 trials, 5 timesteps per trial, 64 features)
    X = X.reshape(12, 5, 64)
    print("X reshaped to (12, 5, 64)")

    # Define label mapping function
    label_to_aksara = {
        '000': 'Aksara A', '001': 'Aksara I', '010': 'Aksara U',
        '011': 'Aksara E', '100': 'Aksara AE', '110': 'Aksara O'
    }

    def enc_lbl(a, b, c):
        key = f"{int(a)}{int(b)}{int(c)}"
        return label_to_aksara.get(key, "Unknown")

    # Combine labels into Aksara labels
    labels = np.array([enc_lbl(y_v[i], y_a[i], y_d[i]) for i in range(len(y_v))])
    print("labels shape:", labels.shape)  # Shape: (60, )

    # Take one label per trial (5 seconds/trial), so 12 labels in total
    labels_per_trial = [labels[i*5] for i in range(12)]
    print("labels_per_trial shape:", np.array(labels_per_trial).shape)

    # Encode labels
    le = LabelEncoder().fit(labels_per_trial)
    y_int = le.transform(labels_per_trial)
    y_cat = to_categorical(y_int)
    print("y_int shape:", y_int.shape)
    print("y_cat shape:", y_cat.shape)

    # Build LSTM model (your spec)
    timesteps, features = X.shape[1], X.shape[2]  # (5, 64)

    model = Sequential([
        Input(shape=(timesteps, features)), # (5, 64)
        LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        LSTM(32, kernel_regularizer=l2(1e-4)), # SCENARIO 2
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(
        loss=CategoricalCrossentropy(),       # multi-class classification
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )

    # Early stopping callback
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Model parameter info (no training changes)
    total_params = model.count_params()
    lstm_params  = sum(layer.count_params() for layer in model.layers if isinstance(layer, LSTM))
    print(f"Model params (total): {total_params:,}")
    print(f"LSTM params only:     {lstm_params:,}")
    print(f"Model config -> timesteps={timesteps}, features={features}, "
          f"lstm_units=64, dropout=0.3, lr=1e-3, l2=1e-4, patience={es.patience}")

    # K-Fold Cross Validation-
    losses, val_losses, val_accs = [], [], []
    kf = KFold(n_splits=10, shuffle=False)
    accs, precs, recs, f1s = [], [], [], []
    all_true, all_pred = [], []
    epochs_used, best_epochs = [], []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y_int), 1):
        print(f"\n=== Fold {fold} ===")

        # Split data
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y_cat[tr_idx], y_cat[te_idx]
        y_test_int = y_int[te_idx]

        # Train model (same model reused across folds as in your original logic)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=8,
            callbacks=[es],
            verbose=0
        )

        # Save metrics from best epoch
        best_epoch = es.stopped_epoch if es.stopped_epoch > 0 else 99
        losses.append(history.history['loss'][best_epoch])
        val_losses.append(history.history['val_loss'][best_epoch])
        accs.append(history.history['accuracy'][best_epoch])
        val_accs.append(history.history['val_accuracy'][best_epoch])

        # Epoch stats
        epochs_used.append(len(history.history['loss']))               
        best_epochs.append(int(np.argmin(history.history['val_loss'])))

        # Prediction and evaluation
        pred = model.predict(X_test, verbose=0).argmax(axis=1)
        all_true.extend(y_test_int)
        all_pred.extend(pred)

        # Additional metrics
        prec = precision_score(y_test_int, pred, average='macro', zero_division=0)
        rec  = recall_score(y_test_int, pred, average='macro', zero_division=0)
        f1   = f1_score(y_test_int, pred, average='macro', zero_division=0)
        precs.append(prec); recs.append(rec); f1s.append(f1)

        print(f"Fold {fold} - Train Acc: {accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")
        print(f"Fold {fold} - Train Loss: {losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Plots (per file)
    hist_df = pd.DataFrame({
        'train_loss': losses,
        'val_loss': val_losses,
        'train_acc': accs,
        'val_acc': val_accs
    }, index=[f"Fold {i+1}" for i in range(len(losses))])

    # Plot loss
    hist_df[['train_loss', 'val_loss']].plot(title=f"Loss per Fold - {input_file}", figsize=(10, 4))
    plt.grid(True); plt.xlabel("Fold"); plt.ylabel("Loss"); plt.show()

    # Plot accuracy
    hist_df[['train_acc', 'val_acc']].plot(title=f"Accuracy per Fold - {input_file}", figsize=(10, 4))
    plt.grid(True); plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.show()

    # Confusion Matrix (per file, aggregated over folds)
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        all_true, all_pred,
        labels=np.arange(len(le.classes_)),
        display_labels=le.classes_,
        cmap='Blues'
    )
    plt.title(f"Confusion Matrix - {input_file}")
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

    # Final metrics (per file)
    print("\n=== FINAL RESULTS (K-Fold CV) ===")
    print(f"Train Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Val Accuracy:   {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    print(f"Train Loss:     {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"Val  Loss:      {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Precision:      {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall:         {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1-score:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Epoch statistics & file summary
    avg_epochs = float(np.mean(epochs_used)) if epochs_used else 0.0
    std_epochs = float(np.std(epochs_used)) if epochs_used else 0.0
    avg_best_ep = float(np.mean(best_epochs)) if best_epochs else 0.0

    print("\n=== EARLY STOPPING / EPOCH STATS ===")
    print(f"Epochs run per fold: {epochs_used}")
    print(f"Average epochs run:  {avg_epochs:.2f} ± {std_epochs:.2f} (patience={es.patience})")
    print(f"Best-epoch (val_loss min) per fold (0-based): {best_epochs}")
    print(f"Average best-epoch index: {avg_best_ep:.2f}")

    print("\n=== FILE / MODEL SUMMARY ===")
    print(f"File: {input_file}")
    print(f"Trials: {X.shape[0]}, Timesteps: {timesteps}, Features: {features}, Classes: {len(le.classes_)}")
    print(f"Params -> LSTM: {lstm_params:,} | Total: {total_params:,}")
    print(f"Optimizer: Adam(lr=1e-3) | Loss: CategoricalCrossentropy | Dropout: 0.3 | L2: 1e-4")

    val_acc_mean, val_acc_std = float(np.mean(val_accs)), float(np.std(val_accs))
    prec_mean, prec_std       = float(np.mean(precs)),   float(np.std(precs))
    rec_mean, rec_std         = float(np.mean(recs)),    float(np.std(recs))
    f1_mean, f1_std           = float(np.mean(f1s)),     float(np.std(f1s))

    print("\n=== PER-FILE SUMMARY ===")
    print(f"File: {input_file}")
    print(f"Accuracy (val):      {val_acc_mean:.4f} ± {val_acc_std:.4f}")
    print(f"Precision (macro):   {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"Recall (macro):      {rec_mean:.4f} ± {rec_std:.4f}")
    print(f"F1-score (macro):    {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Average epochs run:  {avg_epochs:.2f} ± {std_epochs:.2f}")
    print(f"Params -> LSTM: {lstm_params:,} | Total: {total_params:,}")

    # store a compact line for cross-participant summary
    overall.append({
        "file": input_file,
        "val_acc_mean": val_acc_mean,
        "val_acc_std":  val_acc_std,
        "precision_mean": prec_mean,
        "precision_std":  prec_std,
        "recall_mean":    rec_mean,
        "recall_std":     rec_std,
        "f1_mean":        f1_mean,
        "f1_std":         f1_std,
        "epochs_mean":    avg_epochs,
        "epochs_std":     std_epochs,
        "best_epoch_mean": avg_best_ep,
        "classes": int(len(le.classes_)),
        "params_total": int(total_params),
        "params_lstm":  int(lstm_params)
    })

# Summary across all files
if overall:
    summary_df = pd.DataFrame(overall)
    print("\n==================== SUMMARY ACROSS PARTICIPANTS ====================")
    print(summary_df.to_string(index=False))