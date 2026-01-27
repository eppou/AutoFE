import xarray as xr
import os
import numpy as np
from pathlib import Path

# --- CONFIGURAÇÃO ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIR_RAW = PROJECT_ROOT / "data" / "raw" / "clima_parana"
DIR_DEST = PROJECT_ROOT / "data" / "raw" / "clima_parana"
OUT_FILE = DIR_DEST / "PR_precipitacao_2005-2022.nc"

def consertar_chuva():
    print("--- 🌧️ Operação Resgate da Chuva ---")
    
    # 1. Achar o arquivo original
    arquivos = os.listdir(DIR_RAW)
    arquivo_pai = next((f for f in arquivos if f.startswith('pr_')), None)
    
    if not arquivo_pai:
        print("❌ CRÍTICO: Não achei nenhum arquivo começando com 'pr_' na pasta parana!")
        return

    caminho_pai = DIR_RAW / arquivo_pai
    print(f"📁 Arquivo Original encontrado: {arquivo_pai}")
    
    try:
        # 2. Abrir e Inspecionar
        ds = xr.open_dataset(caminho_pai)
        variaveis = list(ds.data_vars)
        print(f"🔎 Variáveis encontradas dentro dele: {variaveis}")
        
        # Tenta adivinhar qual é a chuva
        nome_var_chuva = next((v for v in variaveis if v in ['pr', 'precip', 'tp', 'precipitation']), None)
        
        if not nome_var_chuva:
            print("❌ Não identifiquei qual variável é a chuva. Verifique a lista acima!")
            return
            
        print(f"✅ Variável de chuva identificada: '{nome_var_chuva}'")

        # 3. Processar (Recorte e Renomeação)
        print("✂️ Fazendo recorte do Paraná (2005-2022)...")
        ds_recorte = ds.sel(
            latitude=slice(-22, -28), 
            longitude=slice(-55, -48)
        )
        
        # Filtro de tempo (Segurança extra)
        ds_recorte = ds_recorte.sel(time=slice("2005-01-01", "2022-12-31"))

        # Renomeia para o padrão do nosso projeto
        ds_final = ds_recorte.rename({nome_var_chuva: 'precipitacao'})
        
        # 4. Salvar
        print(f"💾 Salvando arquivo corrigido em: {OUT_FILE}")
        ds_final.to_netcdf(OUT_FILE)
        
        # 5. Validação Imediata
        check = xr.open_dataset(OUT_FILE)
        print("\n--- Validação Final ---")
        print(f"Variáveis no novo arquivo: {list(check.data_vars)}")
        if 'precipitacao' in check.data_vars:
            print("✨ SUCESSO! A chuva foi recuperada.")
        else:
            print("❌ AINDA FALHOU.")

    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")

if __name__ == "__main__":
    consertar_chuva()