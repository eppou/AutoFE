import numpy as np
import pandas as pd
from src.engine import TemporalAggregationEncoder

# ===============================
# 1. Carregar dataset
# ===============================
df = pd.read_parquet("./data/processed/dataset_timeseries_full.parquet")

if 'dias_relativos' not in df.columns:
    raise ValueError("A coluna 'dias_relativos' não foi encontrada no DataFrame.")
# garantir ordenação temporal correta
df = df.sort_values(["id", "dias_relativos"])

# ===============================
# 2. Definir colunas
# ===============================
feature_cols = [
    "Rs", "ETo", "Tmax", "Tmin", "RH", "u2", "pr"
]

target_col = "y"
series_id_col = "id"

WINDOW_SIZE = 181  # tamanho total da série

# ===============================
# 3. Instanciar o encoder
# ===============================
encoder = TemporalAggregationEncoder(
    window_size=WINDOW_SIZE,
    aggregation_windows=[3, 7, 14,21, 30],
    aggregation_functions={
        "mean": np.mean,
        "std": np.std,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
    },
    stride=1,
)

# ===============================
# 4. Construir X e y
# ===============================
X_list = []
y_list = []

for series_id, group in df.groupby(series_id_col):

    # checagem básica
    if len(group) != WINDOW_SIZE:
        continue  # ou raise erro, se quiser rígido

    # matriz temporal [T, num_features]
    series_data = group[feature_cols].values

    # target da série (constante dentro do grupo)
    y_value = group[target_col].iloc[0]

    # aplicar encoder
    X_encoded = encoder.transform(series_data)
    # shape: [T', F]

    X_list.append(X_encoded)
    y_list.append(y_value)

# ===============================
# 5. Converter para arrays
# ===============================
X = np.array(X_list)
y = np.array(y_list)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ===============================
# 6. Salvar em arquivo
# ===============================
np.savez(
    "./data/output/dataset_timeseries_encoded.npz",
    X=X,
    y=y
)

print("Arquivo salvo: dataset_timeseries_encoded.npz")
