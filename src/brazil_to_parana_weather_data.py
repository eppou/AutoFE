import xarray as xr
import os
import glob
import numpy as np

# pega os dados originias disponiveis em https://sites.google.com/site/alexandrecandidoxavierufes/brazilian-daily-weather-gridded-data
# e recorta somente os dados do parana ao inves de usar do brasil todo, economizando armazenamento

# ================= CONFIGURAÇÃO =================
DIR_ENTRADA = "../data/raw/"  # Sugestão: ./data/raw/brasil/
DIR_SAIDA = "../data/raw/clima_parana" # Sugestão: ./data/processed/netcdf/
DATA_INICIO = "2004-01-01"
DATA_FIM = "2022-12-31"

# Bounding Box Paraná
LAT_SLICE = slice(-28.0, -22.0) 
LON_SLICE = slice(-55.0, -48.0)

VARIAVEIS = {
    "pr": "precipitacao",
    "Tmax": "temperatura_max",
    "Tmin": "temperatura_min",
    "RH": "umidade_relativa",
    "Rs": "radiacao_solar",
    "u2": "velocidade_vento",
    "ETo": "evapotranspiracao"
}
# ================================================

def processar_arquivos():
    if not os.path.exists(DIR_SAIDA):
        os.makedirs(DIR_SAIDA)
        
    print(f"--- Iniciando ETL: {DATA_INICIO} a {DATA_FIM} (Paraná) ---\n")

    for prefixo, nome_final in VARIAVEIS.items():
        padrao_busca = os.path.join(DIR_ENTRADA, f"{prefixo}_*.nc")
        print(padrao_busca)
        arquivos_encontrados = glob.glob(padrao_busca)
        
        if not arquivos_encontrados:
            print(f"⚠️  PULANDO: {prefixo} não encontrado.")
            continue
            
        arquivo_bruto = arquivos_encontrados[0]
        print(f"🔄 Processando: {os.path.basename(arquivo_bruto)}...")
        
        try:
            ds = xr.open_dataset(arquivo_bruto)
            print(ds.dims)
            print(ds.coords)
            print(ds.latitude.values[:5], ds.latitude.values[-5:])
            print(ds.longitude.values[:5], ds.longitude.values[-5:])

            

            ds_tempo = ds.sel(time=slice(DATA_INICIO, DATA_FIM))
            ds_recorte = ds_tempo.sel(latitude=LAT_SLICE, longitude=LON_SLICE)

            ds_recorte = ds_recorte.chunk({"time": -1, "latitude": 50, "longitude": 50})
            if ds_recorte.latitude.size == 0:
                ds_recorte = ds_tempo.sel(latitude=slice(-28.0, -22.0), longitude=LON_SLICE)
                
            if ds_recorte.time.size == 0:
                print(f"❌ ERRO: 0 dias selecionados.")
                continue

            # --- MELHORIA: Padronizar coordenadas para evitar erros de float ---
            ds_recorte['latitude'] = np.round(ds_recorte['latitude'], 4)
            ds_recorte['longitude'] = np.round(ds_recorte['longitude'], 4)

            # --- MELHORIA: Compressão para economizar espaço ---
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in ds_recorte.data_vars}

            nome_arquivo = f"PR_{nome_final}_2005-2022.nc"
            caminho_final = os.path.join(DIR_SAIDA, nome_arquivo)
            
            print(f"   💾 Salvando {nome_arquivo}...")
            # compute() não é necessário aqui, o to_netcdf lida com Dask, 
            # mas tirei os chunks na escrita para evitar arquivos fragmentados demais se for ler sequencialmente depois
            ds_recorte.to_netcdf(caminho_final, encoding=encoding) 
            print(f"   ✅ Sucesso!\n")
            
        except Exception as e:
            print(f"❌ ERRO CRÍTICO em {prefixo}: {e}\n")

    print("--- ETL Finalizado! ---")

if __name__ == "__main__":
    processar_arquivos()