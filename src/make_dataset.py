import pandas as pd
import numpy as np
import os
from engine import AutoFE

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
INPUT_FILE = "../data/processed/dataset_timeseries_full.parquet"
OUTPUT_FILE = "../data/output/dataset_lstm_ready_ferrugem.parquet"

# ==============================================================================
# PIPELINE DE GERAÇÃO DE FEATURES (Corrigido para sua classe AutoFE)
# ==============================================================================

# 1. Carregar Dataset
print(f"🔄 Carregando dataset: {INPUT_FILE}")
# Se não tiver o arquivo, crie um dummy para testar o código
if not os.path.exists(INPUT_FILE):
    print("⚠️ Arquivo não encontrado. Criando dados dummy para teste...")
    df = pd.DataFrame({
        'id': [0]*10 + [1]*10,
        'dias_relativos': list(range(-19, -9)) + list(range(-19, -9)),
        'pr': np.random.uniform(0, 20, 20),
        'Tmax': np.random.uniform(20, 35, 20),
        'Tmin': np.random.uniform(15, 25, 20),
        'RH': np.random.uniform(60, 100, 20),
        'Rs': np.random.uniform(10, 30, 20),
        'u2': np.random.uniform(0, 5, 20),
        'ETo': np.random.uniform(2, 6, 20)
    })
else:
    df = pd.read_parquet(INPUT_FILE)

# 2. Pré-processamento básico
# A ferrugem depende da média térmica (Tavg)
df['Tavg'] = (df['Tmax'] + df['Tmin']) / 2

# 3. Instanciar a Lib (Sua classe AutoFE)
fe = AutoFE(id_col='id', time_col='dias_relativos')

print("🚀 Iniciando Feature Engineering...")

# ------------------------------------------------------------------------------
# TIPO 1: DECAIMENTO (Memória de Estado)
# Correção: O nome na sua classe é 'add_decaying_accumulator'
# ------------------------------------------------------------------------------
# Solo úmido (alpha 0.9 = decaimento lento)
df = fe.add_decaying_accumulator(df, 'pr', alpha=0.9, feat_name='soil_moisture_index')

# Estresse UV acumulado (alpha 0.85)
df = fe.add_decaying_accumulator(df, 'Rs', alpha=0.85, feat_name='accumulated_UV_stress')

# ------------------------------------------------------------------------------
# TIPO 2: RECÊNCIA (Forget Gate - Time Since)
# Correção: Passando 'df' como primeiro argumento
# ------------------------------------------------------------------------------
# Chuva forte lava esporos (>15mm)
df['is_heavy_rain'] = df['pr'] >= 15 
df = fe.add_time_since(df, 'is_heavy_rain', 'washoff', fill_na=-1)

# Dias desde umidade crítica (seca prolongada mata o fungo)
df['is_critical_humid'] = df['RH'] >= 90
df = fe.add_time_since(df, 'is_critical_humid', 'critical_humid', fill_na=-1)

# ------------------------------------------------------------------------------
# TIPO 3: PERSISTÊNCIA (Input Gate - Run Length)
# ------------------------------------------------------------------------------
# Sequência de dias molhados (Streak)
df = fe.add_run_length(df, 'is_critical_humid', feat_name='streak_wetness_days')

# Sequência de temperatura ideal (18 a 26.5)
df['is_optimal_temp'] = df['Tavg'].between(18, 26.5)
df = fe.add_run_length(df, 'is_optimal_temp', feat_name='streak_optimal_temp')

# ------------------------------------------------------------------------------
# TIPO 4: INTERAÇÕES (Contexto - Camadas Densas)
# ------------------------------------------------------------------------------
# Índice Termo-Hídrico (Tavg * RH)
df = fe.add_interaction(df, 'Tavg', 'RH', operation='*', normalize=True, feat_name='thermo_hydro_risk_idx')

# Potencial de Secagem (Vento * Evapotranspiração)
df = fe.add_interaction(df, 'u2', 'ETo', operation='*', normalize=True, feat_name='drying_power_idx')

# Z-Score Local de Temperatura (Desvio da média daquele ID específico)
df = fe.add_group_deviation(df, 'Tavg', feat_name='temp_anomaly_local')

# ------------------------------------------------------------------------------
# TIPO 5: CHUVA E RS ACUMULADA (Volume Hídrico)
# ------------------------------------------------------------------------------
print("🛠️ Gerando Chuva Acumulada...")

# Janela de 1 semana (Impacto na esporulação recente)
df = fe.add_rolling_sum(df, 'pr', window=7, feat_name='rain_accum_7d')

# Janela de 1 mês (Impacto no vigor da planta e microclima)
df = fe.add_rolling_sum(df, 'pr', window=30, feat_name='rain_accum_30d')

# DICA EXTRA: Você pode fazer o mesmo para Radiação Solar (Rs)
# "Insolação Acumulada nos últimos 15 dias" diz se o tempo esteve muito aberto ou nublado.
df = fe.add_rolling_sum(df, 'Rs', window=15, feat_name='solar_accum_15d')

# ------------------------------------------------------------------------------
# 5. LIMPEZA E SALVAMENTO
# ------------------------------------------------------------------------------
# Remove colunas auxiliares
cols_to_drop = ['is_heavy_rain', 'is_critical_humid', 'is_optimal_temp']
df.drop(columns=cols_to_drop, inplace=True)

# Ordenação final para LSTM
df = df.sort_values(by=['id', 'dias_relativos'])

print(f"✅ Dataset gerado com sucesso! Colunas: {len(df.columns)}")
# print(df.head()) # Descomente para ver
if os.path.exists(INPUT_FILE):
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"💾 Salvo em: {OUTPUT_FILE}")