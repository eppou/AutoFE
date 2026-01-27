import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm # Barra de progresso (essencial agora que vai demorar mais)

# --- CONFIGURAÇÃO ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_CLIMA = PROJECT_ROOT / "data" / "processed" / "clima_dataset_unificado.parquet"
PATH_TARGET = PROJECT_ROOT / "data" / "processed" / "ocorrencia_dataset.parquet"
PATH_FINAL = PROJECT_ROOT / "data" / "processed" / "dataset_timeseries_full.parquet"

# Quanto tempo para trás queremos guardar? 
# Sugestão: 180 dias cobre com sobra todo o ciclo da soja (R1 a R8 costuma ser < 140 dias)
# O AutoFE poderá cortar janelas menores (30, 60, 90) a partir disso.
MAX_LOOKBACK_DAYS = 180 

def preparar_dataset_timeseries():
    print("--- ⏳ Gerando Dataset Time-Series (Estilo tsfresh) ---")
    
    # 1. Carregar Dados
    print("📂 Carregando arquivos...")
    df_clima = pd.read_parquet(PATH_CLIMA)
    df_target = pd.read_parquet(PATH_TARGET)
    
    # Padronizar nomes
    if 'time' in df_clima.columns: df_clima.rename(columns={'time': 'data'}, inplace=True)
    df_clima['data'] = pd.to_datetime(df_clima['data'])
    df_target['data'] = pd.to_datetime(df_target['data'])
    
    # Criar ID único para cada ocorrência (Essencial para o tsfresh agrupar)
    df_target = df_target.reset_index().rename(columns={'index': 'id_ocorrencia'})
    
    # 2. Match Espacial (KDTree) - O mesmo que funcionou antes
    print("📍 Realizando Match Espacial (Vizinho Mais Próximo)...")
    grid_points = df_clima[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(grid_points[['latitude', 'longitude']].values)
    _, indices = tree.query(df_target[['latitude', 'longitude']].values, k=1)
    
    coords_grid = grid_points.iloc[indices].reset_index(drop=True)
    df_target['lat_grid'] = coords_grid['latitude'].values
    df_target['lon_grid'] = coords_grid['longitude'].values
    
    # 3. O Grande Loop de Extração Temporal
    # Estratégia Otimizada:
    # Em vez de filtrar o climão gigante 5000 vezes, vamos agrupar o clima por local.
    
    print("🚀 Indexando dados climáticos em memória...")
    # Cria um dicionário: chave=(lat, lon) -> valor=DataFrame do Clima daquele ponto
    clima_dict = {
        (lat, lon): grupo.sort_values('data').set_index('data') 
        for (lat, lon), grupo in df_clima.groupby(['latitude', 'longitude'])
    }
    
    print(f"🔄 Explodindo séries temporais (Janela: {MAX_LOOKBACK_DAYS} dias)...")
    lista_series = []
    
    # Itera sobre cada ocorrência de ferrugem
    for _, row in tqdm(df_target.iterrows(), total=len(df_target)):
        
        # Dados do Evento
        evento_id = row['id_ocorrencia']
        data_evento = row['data']
        lat_busca = row['lat_grid']
        lon_busca = row['lon_grid']
        target_val = row['target'] # Dias até infecção (y)
        
        # Busca o histórico climático daquele ponto
        try:
            df_local = clima_dict[(lat_busca, lon_busca)]
            
            # Recorta a Janela Temporal: [Data Evento - 180 dias : Data Evento]
            data_inicio = data_evento - pd.Timedelta(days=MAX_LOOKBACK_DAYS)
            
            # Slice inteligente pelo índice de data
            janela = df_local.loc[data_inicio : data_evento].copy()
            
            if janela.empty:
                continue
                
            # Adiciona colunas de metadados para o tsfresh
            janela['id'] = evento_id
            janela['y'] = target_val
            
            # Cria coluna de tempo relativo (Dia 0 = dia da ocorrência, Dia -1 = ontem...)
            # Isso ajuda muito o tsfresh a entender padrões de "proximidade do evento"
            janela['dias_relativos'] = (janela.index - data_evento).days
            
            # Reseta o índice para a data virar coluna
            janela = janela.reset_index()
            
            # Removemos lat/lon repetidas para economizar espaço (já temos o ID)
            cols_to_drop = ['latitude', 'longitude']
            janela = janela.drop(columns=[c for c in cols_to_drop if c in janela.columns])
            
            lista_series.append(janela)
            
        except KeyError:
            # Caso raríssimo onde o ponto do grid não está no dict (não deve acontecer com KDTree bem feito)
            continue

    # 4. Consolidação Final
    print("💾 Concatenando e salvando...")
    if not lista_series:
        print("❌ Erro: Nenhuma série gerada.")
        return

    df_final = pd.concat(lista_series, ignore_index=True)
    
    print("\n" + "="*40)
    print("       DATASET TIMESERIES GERADO       ")
    print("="*40)
    print(f"🔢 Total de Ocorrências Originais: {len(df_target)}")
    print(f"📈 Total de Linhas (Séries):       {len(df_final)}")
    print(f"📅 Janela Temporal:                {MAX_LOOKBACK_DAYS} dias")
    print(f"📊 Colunas: {list(df_final.columns)}")
    print("="*40)
    
    df_final.to_parquet(PATH_FINAL, index=False)
    print(f"✅ Salvo em: {PATH_FINAL}")

if __name__ == "__main__":
    preparar_dataset_timeseries()