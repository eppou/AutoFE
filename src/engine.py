import pandas as pd
import numpy as np

class BiologicalLSTMFeatureEngineer:
    def __init__(self, df, date_col='data'):
        self.df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # =================================================================
    # 1. Contadores de "Recência" (Time Since) -> Forget Gate (f_t)
    # =================================================================
    def add_time_since_event(self, condition_col, event_name):
        import pandas as pd
import numpy as np

class AutoFE:
    def __init__(self, id_col='id', time_col='dias_relativos'):
        self.id_col = id_col
        self.time_col = time_col

    def add_time_since(self, df: pd.DataFrame, condition_col: str, event_name: str, fill_na=-1):
        """
        Calcula o tempo decorrido desde o último evento, RESPEITANDO a quebra de IDs.
        
        Args:
            df: DataFrame contendo todas as séries.
            condition_col: Nome da coluna booleana ou binária (ex: 'teve_chuva').
            event_name: Nome do sufixo da nova feature.
            fill_na: Valor para preencher quando o evento NUNCA ocorreu naquela série até o momento.
                     (Padrao -1 ou NaN é bom para LSTM mascarar depois).
        """
        # 1. Garantia de Ordenação: Crítico para Time Series
        # Precisamos garantir que os dados estão ordenados por ID e depois por Tempo
        df = df.sort_values(by=[self.id_col, self.time_col]).copy()
        
        # 2. Definição da Máscara do Evento
        mask = df[condition_col].astype(bool)
        
        # 3. Identificar Mudança de Série (ID)
        # O shift(1) compara a linha atual com a anterior. Se for diferente, mudou o ID.
        # fillna(True) garante que a primeira linha seja tratada como mudança.
        id_change_mask = df[self.id_col] != df[self.id_col].shift(1)
        
        # 4. Criação dos "Global Groups" (Reset Triggers)
        # O contador deve resetar se: (Aconteceu o Evento) OU (Mudou o ID)
        reset_triggers = mask | id_change_mask
        print("Reset Triggers:\n", reset_triggers)
        
        # O cumsum cria um ID único para cada segmento de "não-evento" dentro de cada série
        segment_id = reset_triggers.cumsum()
        
        # 5. Contagem dentro do segmento
        # Isso conta 0, 1, 2... dentro de cada bloco sem chuva
        df[f'days_since_{event_name}'] = df.groupby(segment_id).cumcount()
        
        # ==============================================================================
        # CORREÇÃO DE BORDA (O Problema do "Nunca aconteceu ainda")
        # ==============================================================================
        # Se estamos no dia 5 da série ID=10, e nunca choveu na série 10,
        # o contador acima vai mostrar "5". Isso é mentira (bias). Deveria ser NaN ou Infinito.
        
        # Truque: Acumulamos o evento DENTRO do grupo de ID.
        # Se a soma acumulada for 0, significa que para aquele ID, o evento ainda não rolou.
        events_within_id = df.groupby(self.id_col)[condition_col].cumsum()
        
        # Onde events_within_id == 0, o evento nunca aconteceu nessa série.
        # Mas cuidado: se o evento acontece NA LINHA ATUAL, o cumsum é 1, mas o days_since é 0 (correto).
        # O problema é apenas para as linhas ANTES do primeiro evento.
        
        # Lógica refinada: Se (Nunca aconteceu antes) E (Não está acontecendo agora) -> NaN
        undefined_mask = (events_within_id == 0) & (~mask)
        
        df.loc[undefined_mask, f'days_since_{event_name}'] = fill_na
        
        return df

    # =================================================================
    # 2. Acumuladores com Decaimento (Decaying) -> Cell State (C_t)
    # =================================================================
    def add_decaying_accumulator(self, df: pd.DataFrame, value_col: str, alpha: float = 0.9, feat_name: str = None):
        """
        Cria um acumulador com decaimento (ex: Umidade do Solo),
        respeitando a quebra de IDs das séries temporais.
        
        Fórmula Recursiva: y[t] = (1-alpha)*x[t] + alpha*y[t-1]
        (Ou variações dependendo do ajuste do Pandas, aqui usaremos o padrão 'adjust=False' 
         para simular comportamento recursivo físico).
        
        Args:
            df: DataFrame contendo todas as séries.
            value_col: Coluna numérica a ser acumulada (ex: 'precipitacao', 'esporos').
            alpha: Fator de decaimento (0 a 1). 
                   alpha=0.9 significa que retém 90% do valor anterior (decaimento lento).
                   alpha=0.5 significa que retém 50% (decaimento rápido).
            feat_name: Nome opcional. Se None, gera automático.
        """
        # 1. Garantia de Ordenação (Crítico)
        # O cálculo depende da ordem cronológica correta
        df = df.sort_values(by=[self.id_col, self.time_col]).copy()
        
        # 2. Definição do Nome
        if feat_name is None:
            # Remove pontos do float para o nome (0.9 -> 09)
            alpha_str = str(alpha).replace('.', '')
            feat_name = f'decay_{value_col}_{alpha_str}'

        # 3. Função de Aplicação por Grupo
        # O 'transform' aplica a função e retorna uma série indexada igual ao original
        # adjust=False é CRUCIAL para recursão tipo: Hoje + (Ontem * alpha)
        # span, halflife ou alpha podem ser usados. Alpha é mais direto para controle manual.
        
        def calculate_ewm(x):
            return x.ewm(alpha=1-alpha, adjust=False).mean()
        
        # APLICAÇÃO VETORIZADA POR GRUPO
        # Isso isola o ID=0 do ID=1. O final de um não afeta o início do outro.
        df[feat_name] = df.groupby(self.id_col)[value_col].transform(calculate_ewm)
        
        return df

    # =================================================================
    # 3. Janelas de Persistência (Run-Length) -> Input Gate (i_t)
    # =================================================================
    def add_run_length(self, df: pd.DataFrame, condition_col: str, feat_name: str = None):
        """
        Calcula a 'Persistência' (Streak): Quantos dias consecutivos uma condição foi VERDADEIRA.
        Respeita a quebra de IDs (não mistura séries diferentes).
        
        Args:
            df: DataFrame com todas as séries.
            condition_col: Coluna booleana (ex: 'alta_umidade').
            feat_name: Nome da nova feature. Se None, gera automático.
        """
        # 1. Garantia de Ordenação
        df = df.sort_values(by=[self.id_col, self.time_col]).copy()
        
        # 2. Definição do Nome
        if feat_name is None:
            feat_name = f'streak_{condition_col}'
            
        # 3. Preparação da Máscara
        # Garante que é bool. Se for 0/1, vira False/True.
        mask = df[condition_col].astype(bool)
        
        # 4. Detecção de Mudança de Estado (O Segredo da Vetorização)
        # Queremos criar um novo grupo sempre que:
        # A) O ID mudar (mudou de safra/talhão)
        # B) O valor da condição mudar (de True para False ou vice-versa)
        
        # shift(1) pega o valor da linha anterior
        id_changed = df[self.id_col] != df[self.id_col].shift(1)
        condition_changed = mask != mask.shift(1)
        
        # Se qualquer um dos dois mudou, iniciamos um novo "bloco"
        change_flag = id_changed | condition_changed
        
        # cumsum() cria um ID único para cada bloco consecutivo de valores iguais
        streak_id = change_flag.cumsum()
        
        # 5. Contagem Cumulativa dentro do Bloco
        # cumcount() conta 0, 1, 2... dentro do bloco
        # Somamos +1 para que o primeiro dia seja 1 (Intensidade inicial) e não 0.
        streak_count = df.groupby(streak_id).cumcount() + 1
        
        # 6. Aplicação da Máscara
        # O passo anterior conta sequências de True (1, 2, 3) E sequências de False (1, 2, 3).
        # Nós só queremos a persistência do evento (True). Onde for False, deve ser 0.
        df[feat_name] = streak_count * mask.astype(int) # Multiplicação zera onde mask é False
        
        return df

    # =================================================================
    # 4. Interações de Domínio (O "Combo") -> Output Simplification
    # =================================================================
    def add_interaction(self, df: pd.DataFrame, col1: str, col2: str, operation: str = '*', normalize: bool = True, feat_name: str = None):
        """
        Cria uma interação matemática entre duas colunas (O "Combo").
        Ideal para criar índices físicos (ex: Índice Termo-Hídrico).
        
        Args:
            col1, col2: Nomes das colunas.
            operation: '*', '/', '+', '-'.
            normalize: Se True, normaliza (MinMax) as colunas ANTES da operação. 
                       CRUCIAL para LSTM: (0.9 * 0.8) é muito melhor que (25 * 90).
            feat_name: Nome opcional.
        """
        df = df.copy() # Evita SettingWithCopyWarning
        
        # 1. Normalização (Opcional mas recomendada para Deep Learning)
        # Normaliza apenas para o cálculo, não altera o original
        c1 = df[col1]
        c2 = df[col2]
        
        if normalize:
            # MinMax Scaling simples (0 a 1)
            c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-6)
            c2 = (c2 - c2.min()) / (c2.max() - c2.min() + 1e-6)

        # 2. Operação
        if operation == '*':
            res = c1 * c2
            op_symbol = 'x'
        elif operation == '/':
            res = c1 / (c2 + 1e-6) # Proteção contra divisão por zero
            op_symbol = 'div'
        elif operation == '+':
            res = c1 + c2
            op_symbol = 'plus'
        elif operation == '-':
            res = c1 - c2
            op_symbol = 'minus'
        else:
            raise ValueError("Operação não suportada. Use *, /, +, -")

        # 3. Nomeação Automática
        if feat_name is None:
            norm_tag = "norm" if normalize else "raw"
            feat_name = f'inter_{col1}_{op_symbol}_{col2}_{norm_tag}'

        df[feat_name] = res
        return df

    def add_boolean_risk_zone(self, df: pd.DataFrame, conditions: list, feat_name: str):
        """
        Cria uma feature binária (0 ou 1) baseada em múltiplas condições.
        Isso cria 'Gatilhos' explícitos para a LSTM.
        
        Args:
            conditions: Lista de strings com condições pandas query. 
                        Ex: ["temperatura >= 18", "temperatura <= 26", "umidade > 80"]
            feat_name: Nome da nova feature (ex: 'zona_risco_ferrugem').
        """
        # Começa com tudo True
        combined_mask = pd.Series([True] * len(df), index=df.index)
        
        for cond in conditions:
            # Avalia cada condição e combina com AND (&)
            combined_mask = combined_mask & df.eval(cond)
            
        df[feat_name] = combined_mask.astype(int)
        return df

    def add_group_deviation(self, df: pd.DataFrame, target_col: str, feat_name: str = None):
        """
        GROUP-AWARE: Calcula o quanto o valor atual desvia da MÉDIA daquele ID.
        Ajuda a LSTM a entender 'Anomalias Locais' em vez de valores absolutos.
        
        Fórmula: (Valor - Média_do_ID) / Desvio_Padrao_do_ID (Z-Score Local)
        """
        if feat_name is None:
            feat_name = f'zscore_local_{target_col}'
            
        # Calcula média e desvio padrão POR ID (Safra/Local)
        # transform garante que o resultado tenha o mesmo tamanho do df original
        group_mean = df.groupby(self.id_col)[target_col].transform('mean')
        group_std = df.groupby(self.id_col)[target_col].transform('std')
        
        # Z-Score Local (com proteção para std=0)
        df[feat_name] = (df[target_col] - group_mean) / (group_std + 1e-6)
        
        return df
    
    # =================================================================
    # 5. Janelas Móveis (Rolling Statistics) -> Contexto de Volume
    # =================================================================
    def add_rolling_sum(self, df: pd.DataFrame, value_col: str, window: int, feat_name: str = None):
        """
        Calcula a soma acumulada móvel (Rolling Sum), respeitando o ID.
        Ex: Chuva acumulada nos últimos 15 dias.
        """
        # 1. Ordenação Garantida
        df = df.sort_values(by=[self.id_col, self.time_col]).copy()
        
        # 2. Nome Automático
        if feat_name is None:
            feat_name = f'rolling_sum_{value_col}_{window}d'
            
        # 3. Aplicação Vetorizada Group-Aware
        # min_periods=1 garante que não tenhamos NaN no começo da série 
        # (ele soma o que tem disponível até dar a janela completa)
        df[feat_name] = df.groupby(self.id_col)[value_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )
        
        return df

# ==========================================
# Exemplo de Uso da função  1
# ==========================================

# Simulação baseada na sua imagem
data = {
    'id': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1], # Duas séries (0 e 1)
    'dias_relativos': [-180, -179, -178, -177, -176, -175, -174, -180, -179, -178],
    'pr': [0, 16.3, 22.3, 13.3, 4.0, 0, 0, 0, 0, 5.0] # Chuva
}
df = pd.DataFrame(data)

# Criar flag booleana (ex: Chuva Significativa > 5mm)
df['teve_chuva_forte'] = df['pr'] > 5

# Instanciar a Lib
auto_fe = AutoFE(id_col='id', time_col='dias_relativos')

# Calcular
df = auto_fe.add_time_since(df, 'teve_chuva_forte', 'rain_gt_5mm', fill_na=-999)

print(df[['id', 'dias_relativos', 'pr', 'teve_chuva_forte', 'days_since_rain_gt_5mm']])


# ==========================================
# Exemplo de Uso da função  2 
# ==========================================
# Dados simulados: Chuva parando e voltando
data = {
    'id': [0, 0, 0, 0, 0, 1, 1, 1], 
    'dias_relativos': [-175, -174, -173, -172, -171, -175, -174, -173],
    'pr': [10.0, 0.0, 0.0, 20.0, 0.0,  50.0, 0.0, 0.0] 
}
df_test = pd.DataFrame(data)

# Instanciar e Rodar
auto_fe_decay = AutoFE(id_col='id', time_col='dias_relativos')

# Alpha 0.9 (Decaimento Lento - Solo retém água)
df_test = auto_fe_decay.add_decaying_accumulator(df_test, 'pr', alpha=0.9)

# Alpha 0.5 (Decaimento Rápido - Molhamento foliar seca rápido)
df_test = auto_fe_decay.add_decaying_accumulator(df_test, 'pr', alpha=0.5)

print(df_test)

# ==========================================
# Exemplo de Uso da função  3
# ==========================================
data = {
    'id': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    'dias_relativos': [-10, -9, -8, -7, -6, -5, -10, -9, -8, -7],
    'umidade_alta': [False, True, True, True, False, True, True, True, False, True]
}
df_test = pd.DataFrame(data)

# Instanciar
auto_fe_streak = AutoFE(id_col='id', time_col='dias_relativos')

# Calcular
df_test = auto_fe_streak.add_run_length(df_test, 'umidade_alta')

print(df_test)

# ==========================================
# Exemplo de Uso 4
# ==========================================
data = {
    'id': [0, 0, 0, 1, 1, 1],
    'temperatura': [25, 35, 20, 15, 18, 22],  # 35 é muito quente, 18 é ideal
    'umidade':     [90, 40, 85, 95, 90, 80]   # 40 é seco, 90 é úmido
}
df = pd.DataFrame(data)

# Instanciar
auto_fe_combo = AutoFE(id_col='id')

# 1. Interação Multiplicativa (O clássico Thermo-Hydro)
# Normalizar é essencial aqui: (1 * 1) = Risco Máximo.
df = auto_fe_combo.add_interaction(df, 'temperatura', 'umidade', operation='*', normalize=True, feat_name='indice_termo_hidrico')

# 2. Zona de Risco Booleana (O "Combo" Perfeito para Ferrugem)
# Regra: Temp entre 18 e 26 E Umidade > 80
regras_ferrugem = [
    "temperatura >= 18",
    "temperatura <= 26",
    "umidade > 80"
]
df = auto_fe_combo.add_boolean_risk_zone(df, regras_ferrugem, 'risco_biologico_bool')

# 3. Desvio Local (O dia está anormalmente quente PARA ESTE LOCAL?)
df = auto_fe_combo.add_group_deviation(df, 'temperatura')

print(df)