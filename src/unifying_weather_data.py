import xarray as xr
import pandas as pd
import numpy as np
import os
from pathlib import Path

#unifica os varios dados climaticos e passa por um formato tabular (parquet) para facilitar uso


# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIR_PROCESSED = os.path.join(PROJECT_ROOT, "data", "raw","clima_parana")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clima_dataset_unificado.parquet")

def merge_datasets():
    print("--- Iniciando Unificação dos Dados ---")
    
    # 1. Listar todos os arquivos .nc processados
    arquivos = [os.path.join(DIR_PROCESSED, f) for f in os.listdir(DIR_PROCESSED) if f.endswith('.nc')]
    
    if not arquivos:
        print("❌ Nenhum arquivo .nc encontrado em data/processed!")
        return

    try:
        # 2. Abrir múltiplos arquivos e combinar baseados nas coordenadas (time, lat, lon)
        print(f"🔄 Combinando {len(arquivos)} variáveis...")
        ds_combined = xr.open_mfdataset(
            arquivos, 
            combine='by_coords', 
            parallel=False,
            chunks={"time": -1} # Otimização para memória
        )
        
        print("✅ Dados combinados! Dimensões:", ds_combined.sizes)
        
        # 3. (Opcional) Converter para DataFrame se couber na memória
        # Isso facilita muito a manipulação tabular para Machine Learning clássico antes da LSTM
        print("🔄 Convertendo para formato Tabular (Parquet)... isso pode demorar.")
        df = ds_combined.to_dataframe().reset_index().dropna()
        
        # 4. Salvar
        print(f"💾 Salvando em: {OUTPUT_FILE}")
        df.to_parquet(OUTPUT_FILE, index=False)
        
        print("--- Processo Finalizado com Sucesso ---")
        return df

    except Exception as e:
        print(f"❌ Erro na unificação: {e}")

if __name__ == "__main__":
    merge_datasets()