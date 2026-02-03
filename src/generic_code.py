import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. Configurações
# ===============================
NPZ_PATH = "../data/output/dataset_timeseries_encoded.npz"
BATCH_SIZE = 32
EPOCHS = 35
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# 2. Dataset
# ===============================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===============================
# 3. Carregar dados
# ===============================
data = np.load(NPZ_PATH)

X = data["X"]     # [N, T, F]
y = data["y"]     # [N]

# normalização feature-wise
N, T, F = X.shape
X = X.reshape(-1, F)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X.reshape(N, T, F)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_ds = TimeSeriesDataset(X_train, y_train)
test_ds = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ===============================
# 4. Modelo LSTM (baseline igual)
# ===============================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

model = LSTMModel(F).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===============================
# 5. Treinamento
# ===============================
for epoch in range(EPOCHS):
    model.train()
    losses = []

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch+1:02d} | Train MSE: {np.mean(losses):.2f}")

# ===============================
# 6. Avaliação
# ===============================
model.eval()
preds = []
targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        p = model(xb).cpu().numpy()
        preds.extend(p)
        targets.extend(yb.numpy())

preds = np.array(preds)
targets = np.array(targets)

mae = mean_absolute_error(targets, preds)
rmse = root_mean_squared_error(targets, preds)
median_error = np.median(np.abs(targets - preds))

print("\n===== MÉTRICAS LSTM (AUTO-FE) =====")
print(f"MAE (dias): {mae:.2f}")
print(f"RMSE (dias): {rmse:.2f}")
print(f"Erro Mediano (dias): {median_error:.2f}")
