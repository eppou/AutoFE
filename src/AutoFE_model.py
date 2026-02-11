import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# CONFIG
# ===============================
#PARQUET_PATH = "../data/output/dataset_lstm_ready_ferrugem.parquet" #dados com features autogeradas do AutoFE
PARQUET_PATH = "../data/processed/dataset_timeseries_full.parquet" #dados com features originais (sem AutoFE)
ID_COL = "id"
TIME_COL = "dias_relativos"
TARGET_COL = "y"  # dias até ferrugem

SEQ_LEN = 181   # -180 → 0
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# LOAD
# ===============================
df = pd.read_parquet(PARQUET_PATH)
df = df.sort_values([ID_COL, TIME_COL])

#pegua aleatoriamente apenas 30% das series(ids) para acelerar o desenvolvimento
ids = df[ID_COL].unique()
ids_sample = np.random.choice(ids, size=int(len(ids)*0.4), replace=False)
df = df[df[ID_COL].isin(ids_sample)]

# ===============================
# FEATURES
# remove id, tempo e target
# ===============================
ignore_cols = ["id", "y", "data", "safra", "dias_relativos"]

feature_cols = []
for c in df.columns:
    if c in ignore_cols:
        continue
    
    # mantém só numéricas
    if pd.api.types.is_numeric_dtype(df[c]):
        feature_cols.append(c)

print("Features usadas:", feature_cols)
print("Total:", len(feature_cols))

# ===============================
# AGRUPAR POR SÉRIE
# ===============================
X_list = []
y_list = []

for sid, g in df.groupby(ID_COL):
    g = g.sort_values(TIME_COL)

    if len(g) != SEQ_LEN:
        continue

    X_list.append(g[feature_cols].values)
    y_list.append(g[TARGET_COL].iloc[0])

X = np.array(X_list)        # (N, T, F)
y = np.array(y_list)        # (N,)

print("Dataset:", X.shape)

# ===============================
#  SPLIT (Primeiro separamos os dados)
# ===============================
# Fazemos o split nos arrays originais (N, T, F)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
#  NORMALIZAÇÃO (Ajuste fino do Scaler)
# ===============================
N_train, T, F = X_train_raw.shape
N_test, _, _ = X_test_raw.shape

scaler = StandardScaler()

# Achatamos apenas o TREINO para dar o fit
X_train_flat = X_train_raw.reshape(-1, F)
scaler.fit(X_train_flat)  # O scaler APRENDE apenas com o treino

# Transformamos o TREINO
X_train = scaler.transform(X_train_flat).reshape(N_train, T, F)

# Transformamos o TESTE usando a média/desvio do treino
X_test_flat = X_test_raw.reshape(-1, F)
X_test = scaler.transform(X_test_flat).reshape(N_test, T, F)


# ===============================
# DATASET
# ===============================
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(SeqDataset(X_test, y_test), batch_size=BATCH_SIZE)

# ===============================
# MODEL
# ===============================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden=32, layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden,
            layers,
            batch_first=True,
            dropout=0.0
        )

        self.attn = nn.Linear(hidden, 1)

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h, _ = self.lstm(x)      # (B,T,H)
        last = h[:, -1, :]       # último timestep
        return self.head(last).squeeze(-1)

model = LSTMRegressor(F).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ===============================
# TRAIN
# ===============================
for epoch in range(EPOCHS):
    model.train()
    loss_total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    print(f"Epoch {epoch+1}: {loss_total/len(train_loader):.4f}")

# ===============================
# EVAL
# ===============================
model.eval()
preds = []
truth = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb).cpu().numpy()
        preds.extend(out)
        truth.extend(yb.numpy())

preds = np.array(preds)
truth = np.array(truth)

mse = mean_squared_error(truth, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(truth, preds)

print("\n===== MÉTRICAS =====")
print("RMSE:", rmse)
print("MAE :", mae)
print("MSE :", mse)

print(preds.shape, truth.shape)