"""
Training a PyTorch regressor on UCI Combined Cycle Power Plant
Needs internet for first fetch. Saves results in the ./results folder
"""
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo

SEED = 20
np.random.seed(SEED)
torch.manual_seed(SEED)
OUT = Path("results")
OUT.mkdir(exist_ok=True)
#region Data collection and pre-processing
#Downloads public dataset. Four features -> power output in MW.
print("Fetching Combined Cycle Power Plant dataset from UCI...")
dataset = fetch_ucirepo(id=294)
X_df = dataset.data.features
Y_df = dataset.data.targets
print("Features:", list(X_df.columns))
print("Target:", list(Y_df.columns))
print("Rows:", len(X_df))

X = X_df.to_numpy(dtype=np.float32)
y = Y_df.to_numpy(dtype=np.float32).reshape(-1, 1)

#Split data into 70% train, 15% validation, 15% test
#Fit both scalers on training data to avoid leaking validation/test information.
X_train, X_other, y_train, y_other = train_test_split(
    X, y, test_size=0.30, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_other, y_other, test_size=0.50, random_state=SEED
)
scaler_x = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)
X_train_s = scaler_x.transform(X_train).astype(np.float32)
X_val_s = scaler_x.transform(X_val).astype(np.float32)
X_test_s = scaler_x.transform(X_test).astype(np.float32)
y_train_s = scaler_y.transform(y_train).astype(np.float32)
y_val_s = scaler_y.transform(y_val).astype(np.float32)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s)),
    batch_size=128,
    shuffle=True,
)
X_val_t = torch.from_numpy(X_val_s)
y_val_t = torch.from_numpy(y_val_s)
#endregion

#Feed-forward nn
model = nn.Sequential(
    nn.Linear(X.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.MSELoss()

#region Train and check training/validation losses
train_losses, val_losses = [], []
best_loss = float("inf")
best_weights = None
for epoch in range(1, 301):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        predictions = model(xb)
        loss = loss_fn(predictions, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)   # weight by batch size

    train_loss = running_loss / len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val_t), y_val_t).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_loss - 1e-5:
        best_loss = val_loss
        # Copy the best weights so later optimizer steps do not change the saved checkpoint.
        best_weights = deepcopy(model.state_dict())
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}: validation MSE (scaled) = {val_loss:.5f}")
#endregion


model.load_state_dict(best_weights)
model.eval()
with torch.no_grad():
    predicted_scaled = model(torch.from_numpy(X_test_s)).numpy()
# Convert predictions back to MW so errors have physical meaning.
predicted_mw = scaler_y.inverse_transform(predicted_scaled).ravel()
actual_mw = y_test.ravel()

#region Regressor Models
rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train.ravel())
rf_rmse = np.sqrt(mean_squared_error(actual_mw, rf.predict(X_test)))

gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                               max_depth=3, random_state=SEED)
gb.fit(X_train, y_train.ravel())
gb_rmse = np.sqrt(mean_squared_error(actual_mw, gb.predict(X_test)))

print(f"Random forest RMSE = {rf_rmse:.3f} MW")
print(f"Gradient boosting RMSE = {gb_rmse:.3f} MW")
#endregion

# Report errors in original physical units and compare to linear regression.
rmse = np.sqrt(mean_squared_error(actual_mw, predicted_mw))
mae = mean_absolute_error(actual_mw, predicted_mw)
r2 = r2_score(actual_mw, predicted_mw)
linear = LinearRegression().fit(X_train, y_train.ravel())
linear_predictions = linear.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(actual_mw, linear_predictions))
print("\nUnseen TEST data:")
print(f"PyTorch RMSE = {rmse:.3f} MW | MAE = {mae:.3f} MW | R2 = {r2:.3f}")
print(f"Linear regression baseline RMSE = {linear_rmse:.3f} MW")

pd.DataFrame({"actual_MW": actual_mw, "predicted_MW": predicted_mw}).to_csv(
    OUT / "power_plant_predictions.csv", index=False
)

#region Figure Plotting

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(actual_mw, predicted_mw)
lo = min(actual_mw.min(), predicted_mw.min())
hi = max(actual_mw.max(), predicted_mw.max())
ax.plot([lo, hi], [lo, hi], "k--")
ax.set(xlabel="Actual output (MW)", ylabel="Predicted output (MW)",
       title="PyTorch: power-plant predictions on held-out data")
fig.tight_layout()
fig.savefig(OUT / "power_plant_predictions.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="validation")
plt.xlabel("epoch")
plt.ylabel("MSE (scaled)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "loss_curves.png")
plt.show()
#endregion

"""
#region Multi-seed testing

def train_nn(seed, hidden=(32, 16), lr=0.003, weight_decay=0.0,
             use_scheduler=False, max_epochs=300, patience=30):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s)),
        batch_size=128, shuffle=True, generator=gen,
    )

    # Build the network from the list of hidden sizes
    layers, in_dim = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
        if use_scheduler else None
    )
    loss_fn = nn.MSELoss()

    best_loss, best_weights, bad_epochs = float("inf"), None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if scheduler is not None:
            scheduler.step(val_loss)   # lowers the learning rate when val loss stalls

        if val_loss < best_loss - 1e-5:
            best_loss, best_weights, bad_epochs = val_loss, deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.from_numpy(X_test_s)).numpy()
    pred = scaler_y.inverse_transform(pred_s).ravel()
    return np.sqrt(mean_squared_error(actual_mw, pred)), pred
#endregion

#region Multi-seed configs and director
configs = {
    "32-16 (baseline)":       dict(),
    "128-64-32":              dict(hidden=(128, 64, 32)),
    "128-64-32 + decay":      dict(hidden=(128, 64, 32), weight_decay=1e-4),
    "128-64-32 + scheduler":  dict(hidden=(128, 64, 32), use_scheduler=True),
    "128-64-32 + both":       dict(hidden=(128, 64, 32), weight_decay=1e-4, use_scheduler=True),
}

for name, kw in configs.items():
    rmses = []
    for s in range(5):
        rmses.append(train_nn(s, **kw)[0])
        print(f"  {name}, seed {s}: {rmses[-1]:.3f}")
    print(f"{name:24s} {np.mean(rmses):.3f} +- {np.std(rmses):.3f} MW")
#endregion
"""

#region Trying to show residuals of models
#Predictions and residuals for both models, on the same test rows
predictions = {
    "Neural net": predicted_mw,
    "Random forest": rf.predict(X_test),
}
residuals = {name: actual_mw - p for name, p in predictions.items()}
feature_names = list(X_df.columns)

print("Residual summary (MW):")
for name, res in residuals.items():
    print(f"  {name:14s} mean = {res.mean():+.3f}, std = {res.std():.3f}")


def binned_plot(stat, ylabel, filename):
    #One panel per feature; x is split into 10 equal-count bins
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i, ax in enumerate(axes.ravel()):
        bins = pd.qcut(X_test[:, i], q=10, duplicates="drop")
        for name, res in residuals.items():
            binned = pd.Series(res).groupby(bins, observed=True).agg(stat)
            centers = [iv.mid for iv in binned.index]
            ax.plot(centers, binned.values, "-o", label=name)
        if ylabel.startswith("Mean"):
            ax.axhline(0, color="k")
        ax.set(xlabel=feature_names[i], ylabel=ylabel)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.show()


# Bias: does either model systematically over- or under-predict in some range?
binned_plot("mean", "Mean residual (MW)", "residual_bias_by_feature.png")

# Spread: does the error size grow in some range? (RMS of residuals per bin)
rms = lambda r: np.sqrt(np.mean(np.square(r)))
binned_plot(rms, "RMS residual (MW)", "residual_rmse_by_feature.png")

# Do the two models make the same mistakes on the same rows?
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(residuals["Random forest"], residuals["Neural net"])
lim = max(np.abs(residuals["Random forest"]).max(), np.abs(residuals["Neural net"]).max())
ax.plot([-lim, lim], [-lim, lim], "k--")
ax.set(xlabel="Random forest residual (MW)", ylabel="Neural net residual (MW)")
fig.tight_layout()
fig.savefig(OUT / "residual_nn_vs_rf.png")
plt.show()

corr = np.corrcoef(residuals["Random forest"], residuals["Neural net"])[0, 1]
avg_rmse = np.sqrt(mean_squared_error(actual_mw, np.mean(list(predictions.values()), axis=0)))
print(f"Residual correlation (NN vs RF): {corr:.3f}")
print(f"Average-of-both RMSE: {avg_rmse:.3f} MW")
#endregion

