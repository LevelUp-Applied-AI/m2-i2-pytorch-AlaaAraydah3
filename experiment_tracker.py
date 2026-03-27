"""
Stretch Assignment — Experiment Tracker
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import json
import matplotlib.pyplot as plt
from itertools import product

# ─── Model Definition ─────────────────────────────
class HousingModel(nn.Module):
    def __init__(self, hidden_size=32):  
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# ─── Metrics ─────────────────────────────
def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def r2_score(actual, pred):
    ss_res = np.sum((actual - pred)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    return 1 - (ss_res / ss_tot)

# ─── Training Function ─────────────────────────────
def train_model(config, X_train, y_train, X_test, y_test):
    model = HousingModel(hidden_size=config["hidden_size"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    num_epochs = config["epochs"]

    start_time = time.time()
    losses = []

    for epoch in range(num_epochs):
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    training_time = time.time() - start_time

    with torch.no_grad():
        train_pred = model(X_train).numpy().flatten()
        test_pred  = model(X_test).numpy().flatten()
    train_actual = y_train.numpy().flatten()
    test_actual  = y_test.numpy().flatten()

    # Convert float32 → Python float for JSON
    return {
        "config": config,
        "train_loss": float(losses[-1]),
        "test_loss": float(np.mean((test_actual - test_pred)**2)),
        "test_mae": float(mae(test_actual, test_pred)),
        "test_r2": float(r2_score(test_actual, test_pred)),
        "time_s": float(training_time)
    }

# ─── Main Experiment Script ─────────────────────────────
def main():
    df = pd.read_csv("data/housing.csv")
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    # Standardize
    X_scaled = (X - X.mean()) / X.std()
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Shuffle and split
    torch.manual_seed(42)
    indices = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]
    split = int(0.8 * len(X_tensor))
    X_train, X_test = X_shuffled[:split], X_shuffled[split:]
    y_train, y_test = y_shuffled[:split], y_shuffled[split:]

    # Hyperparameter grid
    learning_rates = [0.02, 0.003, 0.004]
    hidden_sizes   = [16, 32, 64]
    num_epochs     = [100, 150, 200]
    configs = [dict(lr=lr, hidden_size=hs, epochs=ep)
               for lr, hs, ep in product(learning_rates, hidden_sizes, num_epochs)]

    all_results = []
    for i, config in enumerate(configs):
        print(f"Running experiment {i+1}/{len(configs)}: {config}")
        result = train_model(config, X_train, y_train, X_test, y_test)
        all_results.append(result)

    # Save JSON
    with open("experiments.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # Leaderboard
    leaderboard = sorted(all_results, key=lambda x: x["test_mae"])
    print("\nLeaderboard (Top 10 by Test MAE):")
    print("Rank | LR      | Hidden | Epochs | Test MAE  | Test R²  | Time (s)")
    print("-----|---------|--------|--------|-----------|----------|--------")
    for rank, res in enumerate(leaderboard[:10], 1):
        c = res["config"]
        print(f"{rank:>4} | {c['lr']:<7} | {c['hidden_size']:<6} | {c['epochs']:<6} | "
              f"{res['test_mae']:<9.2f} | {res['test_r2']:<8.3f} | {res['time_s']:<6.1f}")

    # Summary plot
    plt.figure(figsize=(8,5))
    for hs in hidden_sizes:
        hs_results = [r for r in all_results if r["config"]["hidden_size"] == hs]
        plt.plot([r["config"]["lr"] for r in hs_results],
                 [r["test_mae"] for r in hs_results],
                 marker='o', label=f"Hidden {hs}")
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Test MAE")
    plt.title("Hyperparameter Performance")
    plt.legend()
    plt.savefig("experiment_summary.png")
    plt.close()
    print("Saved experiment_summary.png")

if __name__ == "__main__":
    main()