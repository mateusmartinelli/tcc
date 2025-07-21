import pandas as pd
import numpy as np
import os
import time
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Optional, List, Any
import matplotlib.pyplot as plt
import shutil
import logging
import traceback
from typing import List, Optional, Dict, Tuple
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Diretórios
BASE_DIR = '/home/guest/project'
DATA_DIR = os.path.join(BASE_DIR, 'cointegration_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'cointegration_results', 'distance_adapted')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
MARKET_INDEX_FILE = os.path.join(DATA_DIR, "market_index.csv")

# Arquivos de cache
PAIRS_CACHE_FILE = os.path.join(CACHE_DIR, 'pairs_cache.pkl')
METRICS_CACHE_FILE = os.path.join(CACHE_DIR, 'metrics_cache.pkl')
TRADES_CACHE_FILE = os.path.join(CACHE_DIR, 'trades_cache.pkl')

# Parâmetros da estratégia
TRANSACTION_COST = 0.001  # 0.05% por trade
MIN_SPREAD_STD = 0.002  # Volatilidade mínima do spread
RISK_BUDGET = 0.01  # 1% do capital por trade
STOP_LOSS = 0.07  # 7% stop loss
Z_ENTRY_THRESHOLD = 1.5
Z_EXIT_THRESHOLD_LONG = 0.75
Z_EXIT_THRESHOLD_SHORT = 0.75
MAX_HOLD_DAYS = 50  # Máximo de dias para manter uma posição
LOOKBACKS = [90]  # Período de formação (dias)
ADF_PVALUE_THRESHOLD = 0.10
MAX_TICKERS = 100  # Limite de tickers por período
MIN_TICKERS = 20  # Mínimo de tickers
MIN_PAIRS = 10  # Mínimo de pares cointegrados
MAX_PAIRS = 10  # Máximo de pares a serem negociados
TRADING_DAYS = 183  # Período de trading (6 meses)

TBILL_CSV = os.path.join(DATA_DIR, "Rf.csv")

def load_risk_free() -> pd.Series:
    """
    Carrega a série diária de retornos do T-Bill a partir de Rf.csv.
    Espera colunas 'date' e 'Rf'.
    """
    df = pd.read_csv(TBILL_CSV, parse_dates=['date'])
    df.set_index('date', inplace=True)
    rf = df['Rf'].sort_index()
    return rf

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, list], pd.Series, pd.DataFrame]:
    """Carrega pt_data, rt_data, periods, semester_tickers e rf (risk-free)."""
    start_time = time.time()
    
    # Verificar arquivos necessários
    required = [
        (os.path.join(DATA_DIR, "Pt_cointegration.csv"), "Pt_cointegration.csv"),
        (os.path.join(DATA_DIR, "Rt_cointegration.csv"), "Rt_cointegration.csv"),
        (os.path.join(DATA_DIR, "Rf.csv"),               "Rf.csv"),
        (os.path.join(DATA_DIR, "Periods.csv"),          "Periods.csv"),
        (os.path.join(DATA_DIR, "ticker2.csv"),          "ticker2.csv"),
        (os.path.join(DATA_DIR, "market_index.csv"), "market_index.csv"),
    ]
    for path, name in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    # Carregar preços e retornos
    pt_data = pd.read_csv(os.path.join(DATA_DIR, "Pt_cointegration.csv"), low_memory=False)
    rt_data = pd.read_csv(os.path.join(DATA_DIR, "Rt_cointegration.csv"), low_memory=False)
    periods = pd.read_csv(os.path.join(DATA_DIR, "Periods.csv"), header=None)
    tickers = pd.read_csv(os.path.join(DATA_DIR, "ticker2.csv"))
    
    # Processar pt_data
    pt_data['timestamp'] = pd.to_datetime(pt_data['timestamp'])
    for col in pt_data.columns.difference(['timestamp']):
        pt_data[col] = pd.to_numeric(pt_data[col], errors='coerce')
        if pt_data[col].isna().mean() < 0.20:
            pt_data[col] = pt_data[col].interpolate().ffill().bfill()
        else:
            logging.warning(f"Removendo {col}: NaN ratio={pt_data[col].isna().mean():.2%}")
            pt_data.drop(columns=[col], inplace=True)
    
    # Processar rt_data
    rt_data['timestamp'] = pd.to_datetime(rt_data['timestamp'])
    for col in rt_data.columns.difference(['timestamp']):
        rt_data[col] = pd.to_numeric(rt_data[col], errors='coerce')
    
    # Carregar série risk-free
    rf = load_risk_free()
    
    # Carregar índice de mercado
    market_index = pd.read_csv(MARKET_INDEX_FILE, parse_dates=['Date'])
    market_index.set_index('Date', inplace=True)
    market_index = market_index.sort_index()
    
    # Mapear tickers por semestre
    semester_tickers: Dict[str, list] = {}
    semesters = [(year, half) for year in range(2018, 2025) for half in (1, 2)]
    for idx, col in enumerate(tickers.columns):
        if idx >= len(semesters):
            break
        year, half = semesters[idx]
        key = f"{year}-{'Jan-Jun' if half == 1 else 'Jul-Dec'}"
        start = pd.Timestamp(f"{year}-{'01-01' if half == 1 else '07-01'}")
        end   = pd.Timestamp(f"{year}-{'06-30' if half == 1 else '12-31'}")
        sem_df = pt_data[(pt_data['timestamp'] >= start) & (pt_data['timestamp'] <= end)]
        
        valid = [t for t in tickers[col].dropna().unique() if t in pt_data.columns]
        kept = []
        for t in valid:
            if sem_df[t].isna().mean() < 0.20:
                kept.append(t)
            else:
                logging.warning(f"Ignorando {t} em {key}: NaN={sem_df[t].isna().mean():.2%}")
        semester_tickers[key] = kept
    
    logging.info(f"Dados carregados em {time.time() - start_time:.2f}s")
    return pt_data, rt_data, periods, semester_tickers, rf, market_index

def calculate_ssd(pair: Tuple[str, str], data: pd.DataFrame) -> float:
    """Calcula a soma das diferenças quadradas (SSD) entre os preços normalizados de um par."""
    asset1, asset2 = pair
    prices1 = data[asset1].dropna()
    prices2 = data[asset2].dropna()
    
    # Normalizar preços
    norm_prices1 = prices1 / prices1.iloc[0]
    norm_prices2 = prices2 / prices2.iloc[0]
    
    # Calcular SSD
    ssd = np.sum((norm_prices1 - norm_prices2) ** 2)
    return ssd

def engle_granger_test(y: pd.Series, x: pd.Series) -> Tuple[float, float, float]:
    """Executa o teste de Engle-Granger e retorna p-valor, estatística de teste e hedge ratio."""
    try:
        # Passo 1: Estimar a relação de cointegração
        X = sm.add_constant(x)  # Adiciona constante (intercepto)
        model = OLS(y, X).fit()
        hedge_ratio = model.params.iloc[1]
        spread = y - hedge_ratio * x - model.params.iloc[0]
        
        # Passo 2: Testar estacionariedade do resíduo
        result = adfuller(spread, maxlag=10)
        return result[1], result[0], hedge_ratio
    except Exception as e:
        logging.error(f"Erro no teste Engle-Granger: {e}")
        return 1.0, 0.0, 0.0

def is_integrated_of_order_one(series: pd.Series) -> bool:
    """Verifica se uma série é I(1) usando o teste ADF."""
    try:
        # Testar estacionariedade (I(0))
        adf_result_level = adfuller(series.dropna(), maxlag=10)
        p_value_level = adf_result_level[1]
        
        # Se o nível for estacionário, não é I(1)
        if p_value_level < ADF_PVALUE_THRESHOLD:
            return False
        
        # Testar a primeira diferença para estacionariedade
        diff_series = series.diff().dropna()
        adf_result_diff = adfuller(diff_series, maxlag=10)
        p_value_diff = adf_result_diff[1]
        
        # Se o nível não for estacionário mas a diferença for, é I(1)
        return p_value_diff < ADF_PVALUE_THRESHOLD
    except Exception as e:
        logging.error(f"Erro no teste ADF: {e}")
        return False

def test_pair_cointegration(pair: Tuple[str, str], data: pd.DataFrame) -> Optional[Tuple]:
    asset1, asset2 = pair
    try:
        y1 = data[asset1].dropna()
        x1 = data[asset2].dropna()
        y2 = data[asset2].dropna()
        x2 = data[asset1].dropna()

        best_result = None
        best_pvalue = 1.0

        for (y, x, a1, a2) in [(y1, x1, asset1, asset2), (y2, x2, asset2, asset1)]:
            if not (is_integrated_of_order_one(y) and is_integrated_of_order_one(x)):
                continue

            pvalue, stat, beta = engle_granger_test(y, x)
            if pvalue < ADF_PVALUE_THRESHOLD:
                spread = y - beta * x
                spread_std = spread.std()
                if spread_std >= MIN_SPREAD_STD and pvalue < best_pvalue:
                    best_result = (a1, a2, pvalue, stat, beta, spread_std)
                    best_pvalue = pvalue

        return best_result
    except Exception as e:
        logging.debug(f"Erro ao testar par ({asset1}, {asset2}): {str(e)}")
        return None

def select_cointegrated_pairs(data: pd.DataFrame, lookback: int, valid_tickers: List[str], 
                              pair_cache: Dict, use_cache: bool = True) -> List[Tuple]:
    """Seleciona pares cointegrados com base no SSD e teste de cointegração bidirecional."""
    cache_key = (lookback, tuple(sorted(valid_tickers)))
    if use_cache and cache_key in pair_cache:
        cached_pairs = pair_cache[cache_key]
        if len(cached_pairs) > 0 and len(cached_pairs[0]) == 6:
            logging.info(f"Reutilizando {len(cached_pairs)} pares do cache para lookback {lookback}")
            return cached_pairs
        else:
            logging.warning("Cache inválido - reprocessando")
    
    start_time = time.time()
    assets = [t for t in valid_tickers if t in data.columns and data[t].isna().mean() < 0.20]
    logging.info(f"Período com lookback {lookback}: {len(assets)} ativos válidos")
    
    if len(assets) < MIN_TICKERS:
        logging.warning(f"Ignorando período: apenas {len(assets)} tickers (< {MIN_TICKERS})")
        return []
    
    # 1. Gerar todas as combinações possíveis de pares
    pair_combinations = [(assets[i], assets[j]) for i in range(len(assets)) for j in range(i+1, len(assets))]
    
    # 2. Calcular SSD para cada par
    ssd_results = Parallel(n_jobs=-1)(
        delayed(calculate_ssd)(pair, data.iloc[-lookback:]) for pair in pair_combinations
    )
    
    # 3. Criar dataframe com pares e SSDs
    pairs_df = pd.DataFrame({
        'pair': pair_combinations,
        'ssd': ssd_results
    })

    # 4. Ordenar por SSD (menor primeiro)
    pairs_df = pairs_df.sort_values('ssd').reset_index(drop=True)
    
    # 5. Testar cointegração nas duas ordens para os pares com menor SSD
    cointegrated_pairs = []
    tested_pairs = 0
    max_pairs_to_test = min(100, len(pairs_df))

    for _, row in pairs_df.head(max_pairs_to_test).iterrows():
        pair = row['pair']
        asset1, asset2 = pair
        y1 = data[asset1].dropna()
        x1 = data[asset2].dropna()
        y2 = data[asset2].dropna()
        x2 = data[asset1].dropna()

        best_result = None
        best_pvalue = 1.0

        for (y, x, a1, a2) in [(y1, x1, asset1, asset2), (y2, x2, asset2, asset1)]:
            if not (is_integrated_of_order_one(y) and is_integrated_of_order_one(x)):
                continue

            pvalue, stat, beta = engle_granger_test(y, x)
            if pvalue < ADF_PVALUE_THRESHOLD:
                spread = y - beta * x
                spread_std = spread.std()
                if spread_std >= MIN_SPREAD_STD and pvalue < best_pvalue:
                    best_result = (a1, a2, pvalue, stat, beta, spread_std)
                    best_pvalue = pvalue

        tested_pairs += 1
        if best_result is not None:
            cointegrated_pairs.append(best_result)
            logging.debug(f"Par cointegrado: {best_result[0]}-{best_result[1]} (p={best_result[2]:.4f}, SSD={row['ssd']:.4f})")

            if len(cointegrated_pairs) >= MAX_PAIRS:
                break

    # Função segura para pegar o SSD de qualquer ordem
    def get_ssd(pair, pairs_df):
        cond1 = pairs_df['pair'] == (pair[0], pair[1])
        cond2 = pairs_df['pair'] == (pair[1], pair[0])
        ssd_series = pairs_df[cond1 | cond2]['ssd']
        return ssd_series.values[0] if not ssd_series.empty else np.inf

    # 6. Ordenar pares cointegrados por SSD
    cointegrated_pairs = sorted(cointegrated_pairs, key=lambda x: get_ssd((x[0], x[1]), pairs_df))
    
    logging.info(f"Pares testados: {tested_pairs}, Cointegrados encontrados: {len(cointegrated_pairs)}")
    logging.info(f"Tempo de seleção de pares (lookback={lookback}): {time.time() - start_time:.2f} segundos")
    
    pair_cache[cache_key] = cointegrated_pairs
    return cointegrated_pairs

def compute_spread(pair: Tuple[str, str], data: pd.DataFrame, lookback: int) -> Tuple[pd.Series, float]:
    """Calcula o spread para um par cointegrado via regressão OLS (com constante)."""
    # extrai apenas os dois tickers
    asset1, asset2 = pair[:2]
    # pega os últimos `lookback` pontos e zera o índice
    y = data[asset1].iloc[-lookback:].reset_index(drop=True)
    x = data[asset2].iloc[-lookback:].reset_index(drop=True)

    if y.isna().any() or x.isna().any():
        raise ValueError(f"Dados insuficientes para o par ({asset1}, {asset2})")

    # regressão OLS y = intercept + β·x + u
    X = sm.add_constant(x)
    model = OLS(y, X).fit()
    intercept   = model.params['const']
    hedge_ratio = model.params[asset2]

    # spread “zerado” em média
    spread = y - hedge_ratio * x - intercept
    return spread, hedge_ratio

def calculate_position_size(spread: pd.Series, portfolio_value: float) -> float:
    """Calcula o tamanho da posição com base na volatilidade do spread."""
    spread_vol = np.std(spread)
    if spread_vol == 0:
        return 0
    size = (RISK_BUDGET * portfolio_value) / (spread_vol * Z_ENTRY_THRESHOLD)
    return min(size, portfolio_value)

def trade_pair(pair: Tuple[str, str], data: pd.DataFrame, lookback: int,
              portfolio_value: float, position_state: Dict, trade_log: List, 
              timestamp: datetime) -> Tuple[float, float, int]:
    """Executa a lógica de trading para um par com controle de holding e stop-loss acumulado."""
    pair_key = f"{pair[0]}_{pair[1]}"
    try:
        spread, hedge_ratio = compute_spread(pair, data, lookback)
        z_score = (spread - spread.mean()) / spread.std()
        spread_std = spread.std()
        
        if spread_std < MIN_SPREAD_STD:
            return 0, position_state.get(pair_key, {'position': 0, 'hold_days': 0})['position'], 0
        
        position_size = calculate_position_size(spread, portfolio_value)
        returns = 0
        trade_count = 0
        
        if pair_key not in position_state:
            position_state[pair_key] = {
                'position': 0, 
                'hold_days': 0, 
                'entry_value': 0,
                'asset1_position': 0,
                'asset2_position': 0,
                'entry_date': None,
                'last_trade_date': None
            }
        
        current_z = z_score.iloc[-1]
        current_position = position_state[pair_key]['position']
        hold_days = position_state[pair_key]['hold_days']
        entry_value = position_state[pair_key]['entry_value']
        entry_date = position_state[pair_key]['entry_date']
        last_trade_date = position_state[pair_key]['last_trade_date']
        
        # Calcula dias na posição
        days_in_position = 0
        if entry_date:
            days_in_position = (timestamp - entry_date).days
        
        if current_position != 0:
            position_state[pair_key]['hold_days'] += 1
        
        # Calcular retorno atual do dia
        if current_position > 0:  # Long
            current_return = (spread.iloc[-1] - spread.iloc[-2]) * position_size / portfolio_value
        elif current_position < 0:  # Short
            current_return = -(spread.iloc[-1] - spread.iloc[-2]) * position_size / portfolio_value
        else:
            current_return = 0

        # Calcular retorno acumulado da posição para o stop-loss
        if current_position > 0 and hold_days > 0:  # long spread
            entry_idx = - (hold_days + 1)
            pnl = (spread.iloc[-1] - spread.iloc[entry_idx]) * position_size
            cum_return = pnl / portfolio_value
        elif current_position < 0 and hold_days > 0:  # short spread
            entry_idx = - (hold_days + 1)
            pnl = (spread.iloc[entry_idx] - spread.iloc[-1]) * abs(position_size)
            cum_return = pnl / portfolio_value
        else:
            cum_return = 0
        
        # Lógica de trading
        if current_z < -Z_ENTRY_THRESHOLD and current_position == 0:  # Entrada long
            position_state[pair_key] = {
                'position': position_size,
                'hold_days': 0,
                'entry_value': portfolio_value,
                'asset1_position': position_size,
                'asset2_position': -position_size * hedge_ratio,
                'entry_date': timestamp,
                'last_trade_date': timestamp
            }
            trade_log.append([
                timestamp, pair[0], pair[1], current_z, 0, 'long',
                position_size,
                position_size,
                -position_size * hedge_ratio,
                0  # Dias em operação (acabou de entrar)
            ])
            trade_count = 1
            
        elif current_z > Z_ENTRY_THRESHOLD and current_position == 0:  # Entrada short
            position_state[pair_key] = {
                'position': -position_size,
                'hold_days': 0,
                'entry_value': portfolio_value,
                'asset1_position': -position_size,
                'asset2_position': position_size * hedge_ratio,
                'entry_date': timestamp,
                'last_trade_date': timestamp
            }
            trade_log.append([
                timestamp, pair[0], pair[1], current_z, 0, 'short',
                position_size,
                -position_size,
                position_size * hedge_ratio,
                0  # Dias em operação
            ])
            trade_count = 1

        # Saída Long: critério corrigido para stop-loss acumulado
        elif current_position > 0 and (
            current_z > Z_EXIT_THRESHOLD_LONG or 
            hold_days >= MAX_HOLD_DAYS or
            cum_return <= -STOP_LOSS
        ):
            trade_type = 'close_long'
            if hold_days >= MAX_HOLD_DAYS:
                trade_type = 'max_days'
            elif cum_return <= -STOP_LOSS:
                trade_type = 'stop_loss'
            
            returns = cum_return - TRANSACTION_COST * 2
            trade_log.append([
                timestamp, pair[0], pair[1], current_z, returns, trade_type,
                position_size,
                0,  # Posição zerada
                0,  # Posição zerada
                days_in_position  # Dias em operação
            ])
            position_state[pair_key] = {
                'position': 0,
                'hold_days': 0,
                'entry_value': 0,
                'asset1_position': 0,
                'asset2_position': 0,
                'entry_date': None,
                'last_trade_date': timestamp
            }
            trade_count = 1

        # Saída Short: critério corrigido para stop-loss acumulado
        elif current_position < 0 and (
            current_z < -Z_EXIT_THRESHOLD_SHORT or 
            hold_days >= MAX_HOLD_DAYS or
            cum_return <= -STOP_LOSS
        ):
            trade_type = 'close_short'
            if hold_days >= MAX_HOLD_DAYS:
                trade_type = 'max_days'
            elif cum_return <= -STOP_LOSS:
                trade_type = 'stop_loss'
            
            returns = cum_return - TRANSACTION_COST * 2
            trade_log.append([
                timestamp, pair[0], pair[1], current_z, returns, trade_type,
                position_size,
                0,  # Posição zerada
                0,  # Posição zerada
                days_in_position  # Dias em operação
            ])
            position_state[pair_key] = {
                'position': 0,
                'hold_days': 0,
                'entry_value': 0,
                'asset1_position': 0,
                'asset2_position': 0,
                'entry_date': None,
                'last_trade_date': timestamp
            }
            trade_count = 1
            
        else:  # Manter posição
            returns = current_return
            if current_position != 0:  # Atualiza data do último trade
                position_state[pair_key]['last_trade_date'] = timestamp
        
        return returns, position_state[pair_key]['position'], trade_count
    except Exception as e:
        logging.error(f"Erro ao negociar par {pair}: {e}")
        return 0, position_state.get(pair_key, {'position': 0})['position'], 0
        
def backtest_strategy(pairs: List[Tuple], data: pd.DataFrame, lookback: int,
                     valid_tickers: List[str], period_idx: int, trading_start: datetime) -> Tuple[float, List, int, List]:
    """Realiza backtesting para uma combinação de parâmetros."""
    start_time = time.time()
    portfolio_value = 100000
    portfolio_returns = []
    total_trades = 0
    position_state = {}
    trade_log = []
    
    active_tickers = [t for t in valid_tickers if t in data.columns and data[t].isna().mean() < 0.20]
    trading_length = min(len(data) - lookback, TRADING_DAYS)
    trading_dates = pd.date_range(trading_start, periods=TRADING_DAYS, freq='D')[:trading_length]
    
    logging.info(f"Backtesting com {len(active_tickers)} tickers válidos")
    
    # Validar pares
    active_pairs = []
    for pair in pairs:
        if pair[0] in active_tickers and pair[1] in active_tickers:
            pair_data = data[[pair[0], pair[1]]].iloc[-lookback:]
            if not pair_data.isna().any().any():
                active_pairs.append(pair)
            else:
                logging.warning(f"Par {pair[:2]} ignorado: contém NaNs no lookback")
    
    for t in tqdm(range(lookback, lookback + trading_length), desc=f"Backtesting lookback={lookback}"):
        daily_returns = 0
        daily_trades = 0
        
        for pair in active_pairs:
            ret, pos, trades = trade_pair(
                pair, data.iloc[:t], lookback, portfolio_value,
                position_state, trade_log, data['timestamp'].iloc[t-1]
            )
            daily_returns += ret / max(len(active_pairs), 1)
            daily_trades += trades
        
        portfolio_value *= (1 + daily_returns)
        portfolio_returns.append(daily_returns)
        total_trades += daily_trades
    
    # Pad returns to 183 days
    if len(portfolio_returns) < TRADING_DAYS:
        portfolio_returns.extend([0.0] * (TRADING_DAYS - len(portfolio_returns)))
    
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(365) if np.std(portfolio_returns) > 0 else 0
    logging.info(f"Tempo de backtesting (lookback={lookback}): {time.time() - start_time:.2f} segundos, {total_trades} trades")
    
    return sharpe, portfolio_returns, total_trades, trade_log

def calculate_metrics(
    returns: List[float],
    trade_log: List[Tuple],
    rf_series: pd.Series
) -> Dict:
    """Calcula métricas de desempenho usando série de risk-free, padronizada."""
    # Série de retornos diários
    returns_series = pd.Series(returns)

    # Alinhar rf diário com nossos retornos
    rf_daily = rf_series.reindex(returns_series.index).ffill().fillna(rf_series.mean())

    # Retornos em excesso para métricas de risco
    excess_ret = returns_series - rf_daily

    # --- Retorno cumulativo e annualização geométrica (excesso) ---
    cum_ret = (1 + excess_ret).cumprod() - 1
    if isinstance(returns_series.index, pd.DatetimeIndex) and len(returns_series) > 1:
        days = (returns_series.index[-1] - returns_series.index[0]).days
        years = days / 365
    else:
        years = len(returns_series) / 365 if len(returns_series) > 0 else 0
    ann_ret = (1 + cum_ret.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    # --- Risco anualizado sobre excesso ---
    std_d = excess_ret.std(ddof=1)
    std_a = std_d * np.sqrt(365)

    # --- Sharpe Ratio (excesso) ---
    sharpe = ann_ret / std_a if std_a > 0 else 0

    # --- Drawdowns e drawdown ratios sobre equity bruta ---
    eq_curve    = (1 + returns_series).cumprod()
    rolling_max = eq_curve.cummax()
    drawdowns   = eq_curve / rolling_max - 1
    max_dd      = drawdowns.min()
    avg_dd      = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0

    # --- Sortino Ratio (excesso) ---
    downs    = excess_ret[excess_ret < 0]
    std_down = downs.std(ddof=1) * np.sqrt(365) if not downs.empty else 0
    sortino  = ann_ret / std_down if std_down > 0 else 0

    # --- Omega Ratio (excesso, threshold=0) ---
    gains  = excess_ret[excess_ret > 0].sum()
    losses = -excess_ret[excess_ret < 0].sum()
    omega  = gains / losses if losses > 0 else np.inf

    # --- Kappa-3 Ratio (excesso) ---
    if not downs.empty and (downs**3).mean() > 0:
        dd3    = ((downs**3).mean())**(1/3) * np.sqrt(365)
        kappa3 = ann_ret / dd3 if dd3 > 0 else 0
    else:
        kappa3 = 0

    # --- Calmar & Sterling Ratios ---
    calmar   = ann_ret / abs(max_dd) if max_dd < 0 else np.inf
    sterling = ann_ret / abs(avg_dd) if avg_dd < 0 else np.inf

    # --- Métricas de trade (média aritmética) ---
    trade_df     = pd.DataFrame(trade_log, columns=[
        'timestamp','asset1','asset2','z_score','return',
        'exit_type','position_size','asset1_position',
        'asset2_position','days_in_position'
    ])
    exit_types   = {'close_long','close_short','stop_loss','max_days'}
    exit_df      = trade_df[trade_df['exit_type'].isin(exit_types)]
    num_trades   = len(exit_df)
    win_rate     = (exit_df['return'] > 0).sum() / num_trades if num_trades > 0 else np.nan
    avg_trade_return = exit_df['return'].mean() if num_trades > 0 else np.nan

    return {
        'cumulative_return': cum_ret.iloc[-1],
        'annualized_return': ann_ret,
        'sharpe_ratio':      sharpe,
        'sortino_ratio':     sortino,
        'omega_ratio':       omega,
        'kappa_3_ratio':     kappa3,
        'calmar_ratio':      calmar,
        'sterling_ratio':    sterling,
        'max_drawdown':      max_dd,
        'win_rate':          win_rate,
        'avg_trade_return':  avg_trade_return,
        'num_trades':        num_trades
    }

def identify_crisis_periods(market_index: pd.DataFrame, threshold: float = -0.20) -> pd.DataFrame:
    """
    Identifica períodos de bear market no índice de mercado,
    protegendo contra divisões por zero em picos ou vales.
    """
    # Extrai série de preços
    if isinstance(market_index, pd.DataFrame):
        if 'Close' not in market_index.columns:
            raise ValueError("DataFrame do índice de mercado deve conter coluna 'Close'")
        prices = market_index['Close'].dropna().sort_index()
    else:
        prices = market_index.dropna().sort_index()

    if prices.empty:
        return pd.DataFrame()

    crises = []
    peak = float(prices.iloc[0])
    peak_date = prices.index[0]
    in_crisis = False

    for date, price in prices.items():
        # ignora valores não numéricos
        if not isinstance(price, (float, int)):
            continue

        if not in_crisis:
            # atualiza pico
            if price > peak:
                peak = price
                peak_date = date

            # evita divisão por zero caso o pico seja zero
            if peak == 0:
                logging.warning(f"Pico zero em {date}, pulando cálculo de drawdown")
                continue

            drawdown = (price - peak) / peak
            if drawdown <= threshold:
                in_crisis = True
                start_date = date
                trough = price
                trough_date = date
        else:
            # atualiza vale
            if price < trough:
                trough = price
                trough_date = date

            # evita divisão por zero caso o vale seja zero
            if trough == 0:
                logging.warning(f"Vale zero em {date}, pulando cálculo de recuperação")
                continue

            recovery = (price - trough) / trough
            if recovery >= abs(threshold):
                end_date = date
                crises.append({
                    'start_date':    start_date,
                    'end_date':      end_date,
                    'peak_date':     peak_date,
                    'trough_date':   trough_date,
                    '%_decline':     round((trough - peak) / peak * 100, 2),
                    'duration_days': (end_date - start_date).days + 1
                })
                in_crisis = False
                peak = price
                peak_date = date

    # Se terminou ainda em crise, fecha até o final da série
    if in_crisis:
        end_date = prices.index[-1]
        crises.append({
            'start_date':    start_date,
            'end_date':      end_date,
            'peak_date':     peak_date,
            'trough_date':   trough_date,
            '%_decline':     round((trough - peak) / peak * 100, 2) if peak != 0 else np.nan,
            'duration_days': (end_date - start_date).days + 1
        })

    return pd.DataFrame(crises)

def analyze_crisis_performance(returns: pd.Series,
                               market_index: pd.DataFrame
                              ) -> Dict[str, Any]:
    """
    Analisa o desempenho da estratégia durante períodos de crise e não crise.
    Agora com contagem precisa de dias e verificação de integridade.
    """
    # identifica crises no índice de mercado
    crises = identify_crisis_periods(market_index)
    returns = returns.sort_index()

    # datas totais da série de retornos
    start_date = returns.index.min()
    end_date = returns.index.max()
    total_days = (end_date - start_date).days + 1

    results: Dict[str, Any] = {
        'crisis_periods':   crises,
        'crisis_stats':     None,
        'non_crisis_stats': None,
        'comparison':       None,
        'total_days':       total_days,
        'accounted_days':   0
    }

    # sem crises, só estatísticas gerais
    if crises.empty:
        mean_ret = returns.mean()
        std_ret  = returns.std()
        results['non_crisis_stats'] = {
            'mean_return':   mean_ret,
            'std_return':    std_ret,
            'total_return':  (1 + returns).prod() - 1,
            'sharpe_ratio':  mean_ret / std_ret * np.sqrt(365) if std_ret else np.nan,
            'days':          len(returns)
        }
        results['accounted_days'] = len(returns)
        return results

    # cria máscara de crise para cada dia
    crisis_mask = pd.Series(False, index=returns.index)
    for _, cr in crises.iterrows():
        crisis_mask.loc[cr['start_date']:cr['end_date']] = True

    # estatísticas em crise
    cr_returns = returns[crisis_mask]
    if not cr_returns.empty:
        mean_ret = cr_returns.mean()
        std_ret  = cr_returns.std()
        results['crisis_stats'] = {
            'mean_return':   mean_ret,
            'std_return':    std_ret,
            'total_return':  (1 + cr_returns).prod() - 1,
            'sharpe_ratio':  mean_ret / std_ret * np.sqrt(365) if std_ret else np.nan,
            'days':          len(cr_returns)
        }
        results['accounted_days'] += len(cr_returns)

    # estatísticas fora de crise
    non_cr_returns = returns[~crisis_mask]
    if not non_cr_returns.empty:
        mean_ret = non_cr_returns.mean()
        std_ret  = non_cr_returns.std()
        results['non_crisis_stats'] = {
            'mean_return':   mean_ret,
            'std_return':    std_ret,
            'total_return':  (1 + non_cr_returns).prod() - 1,
            'sharpe_ratio':  mean_ret / std_ret * np.sqrt(365) if std_ret else np.nan,
            'days':          len(non_cr_returns)
        }
        results['accounted_days'] += len(non_cr_returns)

    # valida integridade dos dias
    if results['accounted_days'] != len(returns):
        logging.warning(
            f"Discrepância na contagem de dias: Total={len(returns)}, "
            f"Contabilizados={results['accounted_days']}"
        )

    # comparação retorno e Sharpe
    if results['crisis_stats'] and results['non_crisis_stats']:
        cr_tot = results['crisis_stats']['total_return']
        non_tot = results['non_crisis_stats']['total_return']
        results['comparison'] = {
            'return_ratio':      cr_tot / non_tot if non_tot else np.nan,
            'sharpe_ratio_diff': results['crisis_stats']['sharpe_ratio'] - results['non_crisis_stats']['sharpe_ratio'],
            'outperformance':    cr_tot - non_tot,
            'days_diff':         results['crisis_stats']['days'] - results['non_crisis_stats']['days']
        }

    return results
        
def plot_results(period_metrics: pd.DataFrame, global_metrics: Dict, output_dir: str) -> None:
    import os
    import logging
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    logging.info(f"Entrando em plot_results. output_dir={output_dir}")

    # 1. Garante diretório
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Diretório pronto: {output_dir}")
    except Exception as e:
        logging.error(f"Não foi possível criar diretório: {e}")
        return

    # 2. Prepara série de retornos
    rets = global_metrics.get('daily_returns')
    if rets is None:
        logging.error("daily_returns faltando em global_metrics")
        return

    # Se vier como lista, transforma em Series
    if not isinstance(rets, pd.Series):
        rets = pd.Series(rets)

    # Garante índice datetime
    try:
        rets.index = pd.to_datetime(rets.index)
    except Exception:
        # se não houver índice válido, cria de 1 em 1 dia a partir de 2018-01-01
        rets.index = pd.date_range('2018-01-01', periods=len(rets), freq='D')

    rets = rets.sort_index()
    # Filtro opcional de datas
    rets = rets.loc['2018-01-01':'2024-12-31']
    logging.info(f"Dias de dados para plot: {len(rets)}")
    if rets.empty:
        logging.error("Nenhum dado no intervalo 01/2018–12/2024")
        return

    # 3. Calcula equity e drawdown
    df = pd.DataFrame({'returns': rets})
    df['equity'] = 100 * (1 + df['returns']).cumprod()
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak'] * 100

    # Helper para salvar
    def save_plot(fig, fname):
        path = os.path.join(output_dir, fname)
        try:
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Salvo: {path}")
        except Exception as e:
            logging.error(f"Erro salvando {fname}: {e}")

    # 4. Equity curve
    try:
        logging.info("Plotando equity curve")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df.index, df['equity'], linewidth=2)
        ax.set_title('Evolução do Capital (01/2018–12/2024)')
        ax.set_ylabel('Capital (Base 100)')
        ax.set_xlabel('Data')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(10))
        fig.autofmt_xdate()
        save_plot(fig, 'equity_curve_final.png')
    except Exception as e:
        logging.error(f"Falha ao plotar equity curve: {e}")

    # 5. Drawdown
    try:
        logging.info("Plotando drawdown")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.fill_between(df.index, df['drawdown'], 0, where=df['drawdown'] < 0,
                        color='red', alpha=0.2)
        ax.plot(df.index, df['drawdown'], linewidth=1.5)
        ax.set_title('Drawdown Diário (01/2018–12/2024)')
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Data')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(df['drawdown'].min() * 1.1, 0)
        ax.xaxis.set_major_locator(MaxNLocator(10))
        fig.autofmt_xdate()
        save_plot(fig, 'drawdown_final.png')
    except Exception as e:
        logging.error(f"Falha ao plotar drawdown: {e}")

    # 6. Distribuição de retornos (com zoom central)
    try:
        trade_log = global_metrics.get("trade_log", [])
        if isinstance(trade_log, list) and len(trade_log) > 0:
            trade_columns = [
                'timestamp', 'asset1', 'asset2', 'z_score', 'return', 'exit_type',
                'position_size', 'asset1_position', 'asset2_position', 'days_in_position'
            ]
            trade_df = pd.DataFrame(trade_log, columns=trade_columns)
            exit_types = ["close_long", "close_short", "stop_loss", "max_days"]
            trade_df = trade_df[trade_df["exit_type"].isin(exit_types)]

            if not trade_df.empty:
                trade_returns = trade_df["return"] * 100
                minb = np.floor(trade_returns.min() * 2) / 2
                maxb = np.ceil(trade_returns.max() * 2) / 2
                bins = np.arange(minb, maxb + 0.1, 0.1)

                fig, ax = plt.subplots(figsize=(12, 6))
                n, bins, patches = ax.hist(trade_returns, bins=bins, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Retorno por Trade (%)')
                ax.set_ylabel('Frequência')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(np.arange(minb, maxb + 0.5, 0.5))
                fig.autofmt_xdate()
                save_plot(fig, 'return_distribution_full.png')

                fig, ax = plt.subplots(figsize=(12, 6))
                zoom_threshold = 1.0
                zoom_returns = trade_returns[trade_returns > zoom_threshold]

                if not zoom_returns.empty:
                    minz = np.floor(zoom_returns.min() * 2) / 2
                    maxz = np.ceil(zoom_returns.max() * 2) / 2
                    bins_zoom = np.arange(minz, maxz + 0.1, 0.1)

                    n_zoom, bins_zoom, patches_zoom = ax.hist(zoom_returns, bins=bins_zoom, edgecolor='black', alpha=0.7)
                    ax.set_xlabel('Retorno por Trade (%)')
                    ax.set_ylabel('Frequência')
                    ax.grid(True, alpha=0.3)
                    ax.set_xticks(np.arange(minz, maxz + 0.5, 0.5))

                    for count, bin_left in zip(n_zoom, bins_zoom[:-1]):
                        if count > 0:
                            ax.text(
                                bin_left + (bins_zoom[1] - bins_zoom[0]) / 2,
                                count,
                                f"{int(count)}",
                                ha='center',
                                va='bottom',
                                fontsize=8,
                                rotation=90
                            )
                    fig.autofmt_xdate()
                    save_plot(fig, 'return_distribution_zoom_tail.png')
                else:
                    logging.warning("Sem retornos acima do limiar para plotar cauda.")
            else:
                logging.warning("Nenhum trade encerrado encontrado para plotar zoom na cauda.")
        else:
            logging.warning("trade_log ausente ou vazio; gráfico de zoom na cauda não gerado.")
    except Exception as e:
        logging.error(f"Erro ao plotar distribuição de retornos: {e}")

    # 7. Sharpe Ratio por Período
    try:
        logging.info("Plotando Sharpe por período")
        sharpe_global = global_metrics.get('sharpe_ratio', np.nan)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(period_metrics['period'], period_metrics['sharpe_ratio'],
               width=0.8, edgecolor='black', alpha=0.7)
        ax.axhline(sharpe_global, color='black', linestyle='--', label='Sharpe Global')
        ax.set_title('Sharpe Ratio por Período')
        ax.set_xlabel('Período')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(period_metrics['period'])
        ax.legend()
        fig.autofmt_xdate()
        save_plot(fig, 'sharpe_per_period.png')
    except Exception as e:
        logging.error(f"Falha ao plotar Sharpe por período: {e}")

    # 8. Confere o que foi gerado
    try:
        files = os.listdir(output_dir)
        logging.info(f"Arquivos em {output_dir}: {files}")
    except Exception as e:
        logging.error(f"Não foi possível listar {output_dir}: {e}")

    logging.info("plot_results concluído.")

def generate_rolling_windows(data_start: datetime, data_end: datetime) -> List[Dict]:
    """Gera janelas rolantes com as características especificadas:
    - Formação: 365 dias fixos
    - Trading: 183 dias (6 meses)
    - Passo: 90 dias (3 meses)
    """
    windows = []
    current_start = data_start
    
    while True:
        formation_end = current_start + timedelta(days=365-1)
        trading_end = formation_end + timedelta(days=183)
        
        if trading_end > data_end:
            break
            
        windows.append({
            'formation_start': current_start,
            'formation_end': formation_end,
            'trading_start': formation_end + timedelta(days=1),
            'trading_end': trading_end,
            'period_length': 365,
            'trading_length': 183
        })
        
        current_start += timedelta(days=90)
    
    return windows

def optimize_strategy(periods: pd.DataFrame,
                      pt_data: pd.DataFrame,
                      rt_data: pd.DataFrame,
                      semester_tickers: Dict,
                      rf: pd.Series,
                      use_cache: bool = True,
                      force_reprocess: Optional[List[int]] = None
) -> Tuple[Dict, Dict]:
    """Otimiza parâmetros para cada período com suporte a cache."""
    start_time = time.time()

    # Verificação inicial do cache
    if use_cache:
        cache_available = os.path.exists(PAIRS_CACHE_FILE) and os.path.exists(METRICS_CACHE_FILE)
        if not cache_available:
            logging.warning("Cache não encontrado, processando tudo do zero")
            use_cache = False
        else:
            logging.info("Cache disponível, carregando...")

    # Inicializar caches
    os.makedirs(CACHE_DIR, exist_ok=True)

    if use_cache and os.path.exists(PAIRS_CACHE_FILE):
        try:
            with open(PAIRS_CACHE_FILE, 'rb') as f:
                pair_cache = pickle.load(f)
            logging.info(f"Cache de pares carregado com {len(pair_cache)} entradas")
        except Exception as e:
            logging.error(f"Erro ao carregar cache de pares: {e}")
            pair_cache = {}
    else:
        pair_cache = {}

    if use_cache and os.path.exists(METRICS_CACHE_FILE):
        try:
            with open(METRICS_CACHE_FILE, 'rb') as f:
                metrics_cache = pickle.load(f)
            logging.info(f"Cache de métricas carregado com {len(metrics_cache)} períodos processados")
        except Exception as e:
            logging.error(f"Erro ao carregar cache de métricas: {e}")
            metrics_cache = {}
    else:
        metrics_cache = {}

    # Limpar cache para períodos específicos se necessário
    if force_reprocess and metrics_cache:
        if isinstance(force_reprocess, int):
            force_reprocess = [force_reprocess]

        for period in force_reprocess:
            if f"period_{period}" in metrics_cache:
                metrics_cache.pop(f"period_{period}")
                logging.info(f"Cache limpo para reprocessamento do período {period}")
            else:
                logging.warning(f"Período {period} não encontrado no cache para reprocessamento")

        with open(METRICS_CACHE_FILE, 'wb') as f:
            pickle.dump(metrics_cache, f)

    # Gerar janelas rolantes com as novas características
    data_start = pd.to_datetime("2018-01-01")
    data_end = pd.to_datetime("2024-12-31")
    windows = generate_rolling_windows(data_start, data_end)

    # Preparar para coletar retornos globais
    all_returns = pd.Series(dtype=float)
    all_trade_log = []
    trading_dates = []
    results = []
    processed_periods = 0

    for period_idx, window in enumerate(tqdm(windows, desc="Processando períodos")):
        cache_key = f"period_{period_idx}"

        # Verificar se os resultados estão em cache e não precisam ser reprocessados
        if use_cache and cache_key in metrics_cache and (not force_reprocess or period_idx not in force_reprocess):
            logging.info(f"Usando cache para período {period_idx}")
            result = metrics_cache[cache_key]
            results.append(result)

            # Reconstruir all_returns e all_trade_log a partir do cache
            period_returns = result['returns'][:result['trading_length']]

            period_dates = pd.date_range(
                start=result['trading_start'],
                periods=len(period_returns),
                freq='D'
            )
            if len(period_returns) > 0:
                period_series = pd.Series(period_returns, index=period_dates)
                if not all_returns.empty:
                    all_returns = period_series.combine_first(all_returns)
                else:
                    all_returns = all_returns.combine_first(period_series)
            all_trade_log.extend(result['trade_log'])
            trading_dates.extend(period_dates)
            processed_periods += 1
            continue

        # Processamento do período (quando não está no cache ou precisa ser reprocessado)
        period_start_time = time.time()

        # Determinar semestre para seleção de tickers
        formation_midpoint = window['formation_start'] + timedelta(days=180)
        semester_key = f"{formation_midpoint.year}-{'Jan-Jun' if formation_midpoint.month <= 6 else 'Jul-Dec'}"
        valid_tickers = semester_tickers.get(semester_key, [])

        if len(valid_tickers) < MIN_TICKERS:
            logging.warning(f"Período {period_idx}: Apenas {len(valid_tickers)} tickers (< {MIN_TICKERS})")
            continue

        logging.info(f"Processando período {period_idx}: Formação {window['formation_start'].date()} a {window['formation_end'].date()}, Trading {window['trading_start'].date()} a {window['trading_end'].date()}")

        formation_data = pt_data[(pt_data['timestamp'] >= window['formation_start']) &
                               (pt_data['timestamp'] <= window['formation_end'])]
        if len(formation_data) < max(LOOKBACKS):
            logging.warning(f"Período {period_idx} ignorado: dados insuficientes ({len(formation_data)} linhas)")
            continue

        # Selecionar pares cointegrados com base no SSD
        pairs = []
        for lookback in LOOKBACKS:
            period_pairs = select_cointegrated_pairs(formation_data, lookback, valid_tickers, pair_cache, use_cache)
            pairs.extend(period_pairs)

        # Manter apenas os MAX_PAIRS pares com menor SSD
        pairs = sorted(pairs, key=lambda x: calculate_ssd((x[0], x[1]), formation_data.iloc[-lookback:]))[:MAX_PAIRS]

        # Salvar pares
        pairs_df = pd.DataFrame(pairs, columns=['asset1', 'asset2', 'eg_pvalue', 'eg_stat', 'hedge_ratio', 'spread_std'])
        pairs_df.to_csv(os.path.join(RESULTS_DIR, f"pairs_period_{period_idx}.csv"), index=False)
        logging.info(f"Período {period_idx}: {len(pairs)} pares cointegrados selecionados (menor SSD)")

        if len(pairs) < MIN_PAIRS:
            logging.warning(f"Período {period_idx} ignorado: apenas {len(pairs)} pares (< {MIN_PAIRS})")
            continue

        trading_data = pt_data[(pt_data['timestamp'] >= window['trading_start']) &
                             (pt_data['timestamp'] <= window['trading_end'])]
        if len(trading_data) < 10:
            logging.warning(f"Período {period_idx} ignorado: dados de trading insuficientes ({len(trading_data)} linhas)")
            continue

        # Backtesting
        period_trades = 0
        period_returns = []
        best_sharpe = -np.inf
        best_result = None

        for lookback in LOOKBACKS:
            sharpe, returns, trades, trade_log = backtest_strategy(
                pairs, trading_data, lookback, valid_tickers, period_idx, window['trading_start']
            )

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_result = {
                    'period': period_idx,
                    'lookback': lookback,
                    'sharpe': sharpe,
                    'returns': returns,
                    'trade_log': trade_log,
                    'metrics': calculate_metrics(returns, trade_log, rf),
                    'formation_start': window['formation_start'],
                    'formation_end': window['formation_end'],
                    'trading_start': window['trading_start'],
                    'trading_end': window['trading_end'],
                    'trading_length': window['trading_length']
                }

            period_trades += trades
            period_returns.extend(returns)

        if best_result is None:
            continue

        # Armazenar resultados
        results.append(best_result)
        metrics_cache[cache_key] = best_result
        processed_periods += 1

        # Coletar retornos para análise global
        period_dates = pd.date_range(window['trading_start'], periods=len(period_returns), freq='D')
        if len(period_returns) > 0:
            period_series = pd.Series(period_returns, index=period_dates)
            if not all_returns.empty:
                all_returns = pd.concat([all_returns, period_series])
            else:
                all_returns = period_series
        trading_dates.extend(period_dates)
        all_trade_log.extend(best_result['trade_log'])

        # Salvar cache periodicamente
        if period_idx % 5 == 0 or period_idx == len(windows) - 1:
            try:
                with open(METRICS_CACHE_FILE, 'wb') as f:
                    pickle.dump(metrics_cache, f)
                with open(PAIRS_CACHE_FILE, 'wb') as f:
                    pickle.dump(pair_cache, f)
                logging.info(f"Cache salvo após período {period_idx}")
            except Exception as e:
                logging.error(f"Erro ao salvar cache: {e}")

        logging.info(f"Período {period_idx} concluído em {time.time() - period_start_time:.2f}s. Trades: {period_trades}, Sharpe: {best_sharpe:.4f}")

    # Transformar results em best_results
    best_results = {}
    for result in results:
        period_idx = result['period']
        best_results[period_idx] = result

        # Salvar returns usando trading_length do resultado
        trading_length = result['trading_length']
        returns_df = pd.DataFrame(result['returns'][:trading_length], columns=['returns'])
        returns_df.to_csv(os.path.join(RESULTS_DIR, f"returns_period_{period_idx}.csv"), index=False)

    # Verificação de períodos duplicados
    period_ranges = [(r['formation_start'], r['trading_end']) for r in results]
    if len(period_ranges) != len(set(period_ranges)):
        logging.warning("Atenção: períodos duplicados detectados!")

    # Calcular métricas globais
    if not all_returns.empty:
        crisis_periods = identify_crisis_periods(all_returns)
        global_metrics = calculate_metrics(all_returns.tolist(), all_trade_log, rf)
        global_metrics.update({
            'daily_returns': all_returns,
            'cumulative_returns': (1 + all_returns).cumprod() - 1,
            'drawdowns': ((1 + all_returns).cumprod() / (1 + all_returns).cumprod().cummax() - 1),
            'crisis_periods': crisis_periods
        })
    else:
        global_metrics = {
            'daily_returns': pd.Series(dtype=float),
            'cumulative_returns': pd.Series(dtype=float),
            'drawdowns': pd.Series(dtype=float),
            'crisis_periods': pd.DataFrame()
        }
        logging.warning("Nenhum retorno válido para cálculo de métricas globais")

    # Inclui trade_log nas métricas globais
    global_metrics['trade_log'] = all_trade_log

    # Salvar métricas por período
    period_metrics = []
    for period_idx, result in best_results.items():
        pr = pd.Series(result['returns'][:result['trading_length']])
        period_return = (1 + pr).prod() - 1 if not pr.empty else 0.0
        period_metrics.append({
            'period': period_idx,
            'lookback': result['lookback'],
            'sharpe_ratio': result['sharpe'],
            'cumulative_return': result['metrics']['cumulative_return'],
            'period_return': period_return,
            'num_trades': result['metrics']['num_trades'],
            'win_rate': result['metrics']['win_rate'],
            'avg_trade_return': result['metrics']['avg_trade_return']
        })

    metrics_df = pd.DataFrame(period_metrics)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)

    # Gerar gráficos
    try:
        plot_results(metrics_df, global_metrics, ANALYSIS_DIR)
        logging.info("Gráficos gerados com sucesso")
    except Exception as e:
        logging.error(f"Erro ao gerar gráficos: {e}")

    # Salvar períodos de crise, se existirem
    if 'crisis_periods' in global_metrics and not global_metrics['crisis_periods'].empty:
        crisis_file = os.path.join(RESULTS_DIR, "crisis_periods.csv")
        global_metrics['crisis_periods'].to_csv(crisis_file, index=False)
        logging.info(f"Períodos de crise salvos em {crisis_file}")

    logging.info(f"Processamento concluído. {processed_periods}/{len(windows)} períodos processados. Tempo total: {time.time() - start_time:.2f} segundos")
    return best_results, global_metrics

def main(use_cache: bool = True, force_reprocess: Optional[List[int]] = None) -> None:
    """Função principal com todas as atualizações."""
    try:
        start_time = time.time()

        # 1) Limpa resultados se não usar cache
        if not use_cache and os.path.exists(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR)

        # 2) Garante diretórios
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

        logging.info(f"Modo cache {'ativado' if use_cache else 'desativado'}.")

        # 3) Carrega dados
        pt_data, rt_data, periods, semester_tickers, rf_series, market_index = load_data()

        # 4) Executa otimização (corrigido: sem passar market_index)
        best_results, global_metrics = optimize_strategy(
            periods,
            pt_data,
            rt_data,
            semester_tickers,
            rf_series,
            use_cache=use_cache,
            force_reprocess=force_reprocess
        )

        # 5) Assegura formatos de daily_returns para plotting
        all_returns = global_metrics.get('daily_returns')
        if isinstance(all_returns, pd.Series):
            global_metrics['daily_returns'] = all_returns.values
            global_metrics['daily_returns_index'] = all_returns.index
        else:
            idx = global_metrics.get('daily_returns_index')
            if idx is None:
                raise ValueError("Falta 'daily_returns_index' em global_metrics.")
            global_metrics['daily_returns'] = all_returns
            global_metrics['daily_returns_index'] = idx

        # 6) Analisa desempenho em crises usando o índice de mercado
        returns_series = pd.Series(
            global_metrics['daily_returns'],
            index=global_metrics['daily_returns_index']
        )
        crisis_res = analyze_crisis_performance(returns_series, market_index)
        global_metrics.update(crisis_res)

        # 7) Log de resultados de crise
        cs  = global_metrics.get('crisis_stats', {})
        ncs = global_metrics.get('non_crisis_stats', {})
        cmp = global_metrics.get('comparison', {})
        if cs:
            logging.info("\nDesempenho Durante Crises:")
            logging.info(f"Dias em crise           : {cs.get('days', 0)}")
            logging.info(f"Retorno total crise     : {cs.get('total_return', 0)*100:.2f}%")
            logging.info(f"Sharpe Ratio crise      : {cs.get('sharpe_ratio', 0):.2f}")
        if ncs:
            logging.info("\nDesempenho Fora de Crises:")
            logging.info(f"Dias fora de crise      : {ncs.get('days', 0)}")
            logging.info(f"Retorno total fora      : {ncs.get('total_return', 0)*100:.2f}%")
            logging.info(f"Sharpe Ratio fora       : {ncs.get('sharpe_ratio', 0):.2f}")
        if cmp:
            logging.info("\nComparação:")
            logging.info(f"Retorno relativo (c/n)  : {cmp.get('return_ratio', 0):.2f}")
            logging.info(f"Diferença Sharpe        : {cmp.get('sharpe_ratio_diff', 0):.2f}")
            logging.info(f"Outperformance absoluta : {cmp.get('outperformance', 0)*100:.2f}%")

        # 8) Exibe métricas globais
        logging.info("\nMétricas Globais:")
        for key, label, fmt in [
            ('cumulative_return',  'Retorno Cumulativo     ', '{:.2f}%'),
            ('annualized_return',  'Retorno Anualizado     ', '{:.2f}%'),
            ('sharpe_ratio',       'Sharpe Ratio           ', '{:.4f}'),
            ('sortino_ratio',      'Sortino Ratio          ', '{:.4f}'),
            ('omega_ratio',        'Omega Ratio            ', '{:.4f}'),
            ('kappa_3_ratio',      'Kappa 3 Ratio          ', '{:.4f}'),
            ('calmar_ratio',       'Calmar Ratio           ', '{:.4f}'),
            ('sterling_ratio',     'Sterling Ratio         ', '{:.4f}'),
            ('max_drawdown',       'Drawdown Máximo        ', '{:.2f}%'),
            ('win_rate',           'Taxa de Vitórias       ', '{:.2f}%'),
            ('avg_trade_return',   'Retorno Médio por Trade', '{:.4f}%'),
            ('num_trades',         'Total de Trades        ', '{}')
        ]:
            val = global_metrics.get(key, 0)
            if key in ['cumulative_return', 'annualized_return',
                       'max_drawdown', 'win_rate', 'avg_trade_return']:
                val *= 100
            logging.info(f"{label}: {fmt.format(val)}")

        # 9) Exibe métricas por período
        logging.info("\nMétricas por Período:")
        for res in best_results.values():
            start = res['trading_start'].strftime('%d/%m/%Y')
            end   = res['trading_end'].strftime('%d/%m/%Y')
            logging.info(
                f"Período {start} a {end}: "
                f"Lookback={res['lookback']}, "
                f"Sharpe={res['sharpe']:.4f}, "
                f"Retorno={res['metrics']['cumulative_return']*100:.2f}%, "
                f"Trades={res['metrics']['num_trades']}, "
                f"Win Rate={res['metrics']['win_rate']*100:.2f}%"
            )

        # 10) Geração de gráficos e arquivos de saída
        try:
            period_data = [{
                'period':        idx,
                'lookback':      res['lookback'],
                'sharpe_ratio':  res['sharpe'],
                'cumulative_return': res['metrics']['cumulative_return'],
                'num_trades':    res['metrics']['num_trades'],
                'win_rate':      res['metrics']['win_rate'],
                'start_date':    res['trading_start'],
                'end_date':      res['trading_end']
            } for idx, res in best_results.items()]

            period_metrics = pd.DataFrame(period_data).sort_values('period')
            period_metrics.to_csv(os.path.join(ANALYSIS_DIR, "period_metrics.csv"), index=False)

            plot_results(period_metrics, global_metrics, ANALYSIS_DIR)

            if 'trade_log' in global_metrics:
                save_trades_to_csv(global_metrics['trade_log'], ANALYSIS_DIR)

            logging.info(f"Arquivos de análise gerados em {ANALYSIS_DIR}")
        except Exception as e:
            logging.error(f"Falha ao gerar saídas: {e}\n{traceback.format_exc()}")

    except FileNotFoundError as e:
        logging.error(f"Erro: {e}")
    except ValueError as e:
        logging.error(f"Erro de valor: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}\n{traceback.format_exc()}")
if __name__ == "__main__":
    # Clear old cache if needed
    if os.path.exists(PAIRS_CACHE_FILE):
        os.remove(PAIRS_CACHE_FILE)
    if os.path.exists(METRICS_CACHE_FILE):
        os.remove(METRICS_CACHE_FILE)
    
    # Run with new parameters
    main(use_cache=False)
    
    # After first run, you can use cache
