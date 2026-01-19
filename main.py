# main.py
import argparse
import yaml
import logging
from src.data_loader import load_agronomic_data
from src.engine import AutoFEModel
from src.competitors import RandomForestBaseline, XGBoostBaseline
from src.utils import save_results, setup_logger

# Configuração do Logger
logger = setup_logger()

def get_model(model_name, config):
    """Factory Pattern para instanciar o modelo correto"""
    if model_name == 'autofe':
        logger.info("Inicializando Auto Learned Features Model (LSTM)...")
        return AutoFEModel(config['autofe_params'])
    
    elif model_name == 'rf':
        logger.info("Inicializando Random Forest Baseline...")
        return RandomForestBaseline(config['rf_params'])
    
    elif model_name == 'xgboost':
        logger.info("Inicializando XGBoost Baseline...")
        return XGBoostBaseline(config['xgb_params'])
    
    else:
        raise ValueError(f"Modelo '{model_name}' não reconhecido.")

def main(args):
    # 1. Carregar Configurações
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Iniciando pipeline para: {args.model.upper()}")

    # 2. Carregar Dados (Inputs)
    # Aqui você carrega seus dados de soja/agronômicos
    X_train, X_test, y_train, y_test = load_agronomic_data(
        path=config['data_path'], 
        target=config['target_col']
    )

    # 3. Instanciar Modelo
    model = get_model(args.model, config)

    # 4. Pipeline de Treinamento
    logger.info("Iniciando treinamento...")
    model.train(X_train, y_train)

    # 5. Pipeline de Avaliação
    logger.info("Iniciando avaliação...")
    metrics = model.evaluate(X_test, y_test)
    
    # 6. Salvar Outputs (Resultados e o Modelo em si)
    save_results(metrics, args.model, output_dir='data/outputs')
    if args.save_model:
        model.save(f"models/{args.model}_latest.pkl")
    
    logger.info(f"Processo finalizado. Acurácia/Loss: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Execução - AutoFE Project")
    
    parser.add_argument('--model', type=str, required=True, 
                        choices=['autofe', 'rf', 'xgboost'],
                        help='Escolha o modelo para rodar: autofe (seu projeto), rf, ou xgboost')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Caminho para o arquivo de configuração YAML')
    
    parser.add_argument('--save_model', action='store_true',
                        help='Flag para salvar o binário do modelo treinado')

    args = parser.parse_args()
    main(args)