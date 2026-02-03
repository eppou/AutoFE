import pandas as pd
import numpy as np

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# 1. Ler parquet
# ===============================
df = pd.read_parquet("../data/processed/dataset_timeseries_full.parquet")

# ordenar corretamente
df = df.sort_values(["id", "dias_relativos"])

# manter janela fixa
df = df[df["dias_relativos"].between(-180, -1)]

# ===============================
# 2. Definir colunas
# ===============================
feature_cols = ["Rs", "ETo", "Tmax", "Tmin", "RH", "u2", "pr"]
target_col = "y"

# tsfresh espera:
# id -> série
# time -> ordem temporal
tsfresh_df = df[["id", "dias_relativos"] + feature_cols]

# ===============================
# 3. Extrair features com tsfresh
# ===============================
fc_parameters = MinimalFCParameters()
X = extract_features(
    tsfresh_df,
    column_id="id",
    column_sort="dias_relativos",
    disable_progressbar=False,
    n_jobs=8,
    default_fc_parameters=fc_parameters
)

# preencher NaNs gerados
impute(X)

# ===============================
# 4. Alvo (1 y por id)
# ===============================
y = (
    df.groupby("id")[target_col]
      .first()
      .loc[X.index]
      .values
)

print("Features tsfresh:", X.shape)

# ===============================
# 5. Split treino / teste
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 6. Random Forest
# ===============================
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# ===============================
# 7. Avaliação
# ===============================
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

acc_5 = np.mean(np.abs(y_pred - y_test) <= 5)
acc_10 = np.mean(np.abs(y_pred - y_test) <= 10)

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f} dias")
print(f"MAE:  {mae:.2f} dias")
print(f"Acc@5d:  {acc_5*100:.1f}%")
print(f"Acc@10d: {acc_10*100:.1f}%")
