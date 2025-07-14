import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import itertools
import os
import time
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import shutil
import logging
import traceback
import statsmodels.api as sm
from typing import List, Optional, Dict, Tuple
from typing import Union, List, Tuple, Dict, Any

# Configurações de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurações
COINTEGRATION_DATA_DIR = "/home/guest/project/cointegration_data"
DISTANCE_DATA_DIR = "/home/guest/project/distance_data"
RESULTS_DIR = "/home/guest/project/cointegration_results/threshold_1"
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
MARKET_INDEX_FILE = os.path.join(COINTEGRATION_DATA_DIR, "market_index.csv")

# Arquivos de cache
PAIRS_CACHE_FILE = os.path.join(CACHE_DIR, "pairs_cache.pkl")
METRICS_CACHE_FILE = os.path.join(CACHE_DIR, "metrics_cache.pkl")
TRADES_CACHE_FILE = os.path.join(CACHE_DIR, "trades_cache.pkl")

# Parâmetros da estratégia
TRANSACTION_COST = 0.001  # 0.05% por trade
MIN_SPREAD_STD = 0.002  # Volatilidade mínima do spread
RISK_BUDGET = 0.01  # 1% do capital por trade
STOP_LOSS = 0.07  # 7% stop loss
Z_ENTRY_THRESHOLD = 1.5
Z_EXIT_THRESHOLD_LONG = 0.75
Z_EXIT_THRESHOLD_SHORT = 0.75
MAX_HOLD_DAYS = 50  # Máximo de dias para manter uma posição
LOOKBACKS = [90]
ADF_PVALUE_THRESHOLD = 0.10
MAX_TICKERS = 100  # Limite de tickers por período
MIN_TICKERS = 20  # Mínimo de tickers
MIN_PAIRS = 10  # Reduzido de 5 para 3
MAX_PAIRS = 10
TRADING_DAYS = 183

TBILL_CSV = os.path.join(COINTEGRATION_DATA_DIR, "Rf.csv")

def load_risk_free() -> pd.Series:
    """
    Carrega a série de retornos diários do T-Bill e devolve um pd.Series indexada por data.
    Espera colunas 'date' e 'Rf' no CSV.
    """
    df = pd.read_csv(TBILL_CSV, parse_dates=['date'])
    df.set_index('date', inplace=True)
    rf = df['Rf'].sort_index()  # a coluna no seu CSV chama-se "Rf"
    return rf

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, list], pd.Series, pd.DataFrame]:
    """Carrega pt_data, rt_data, periods, semester_tickers e rf (risk-free)."""
    start_time = time.time()
    
    # Verificar arquivos necessários
    required = [
        (os.path.join(COINTEGRATION_DATA_DIR, "Pt_cointegration.csv"), "Pt_cointegration.csv"),
        (os.path.join(COINTEGRATION_DATA_DIR, "Rt_cointegration.csv"), "Rt_cointegration.csv"),
        (os.path.join(COINTEGRATION_DATA_DIR, "Rf.csv"),               "Rf.csv"),
        (os.path.join(DISTANCE_DATA_DIR, "Periods.csv"),              "Periods.csv"),
        (os.path.join(DISTANCE_DATA_DIR, "ticker2.csv"),              "ticker2.csv"),
        (os.path.join(COINTEGRATION_DATA_DIR, "market_index.csv"), "market_index.csv"),
    ]
    for path, name in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    # Carregar preços e retornos
    pt_data = pd.read_csv(os.path.join(COINTEGRATION_DATA_DIR, "Pt_cointegration.csv"), low_memory=False)
    rt_data = pd.read_csv(os.path.join(COINTEGRATION_DATA_DIR, "Rt_cointegration.csv"), low_memory=False)
    periods = pd.read_csv(os.path.join(DISTANCE_DATA_DIR, "Periods.csv"), header=None)
    tickers = pd.read_csv(os.path.join(DISTANCE_DATA_DIR, "ticker2.csv"))
    
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
            if sem_df[t].isna().mean() < 0.05:
                kept.append(t)
            else:
                logging.warning(f"Ignorando {t} em {key}: NaN={sem_df[t].isna().mean():.2%}")
        semester_tickers[key] = kept
    
    logging.info(f"Dados carregados em {time.time() - start_time:.2f}s")
    return pt_data, rt_data, periods, semester_tickers, rf, market_index

def engle_granger_test(y: pd.Series, x: pd.Series) -> Tuple[float, float]:
    """Executa o teste de Engle-Granger e retorna p-valor e estatística de teste."""
    try:
        # Passo 1: Estimar a relação de cointegração
        model = OLS(y, sm.add_constant(x)).fit()
        spread = y - model.params.iloc[1] * x - model.params.iloc[0]  # Usando iloc
        
        # Passo 2: Testar estacionariedade do resíduo
        result = adfuller(spread, maxlag=10)
        return result[1], result[0]
    except Exception as e:
        logging.error(f"Erro no teste Engle-Granger: {e}")
        return 1.0, 0.0

def johansen_test(y: pd.Series, x: pd.Series) -> Tuple[float, float]:
    """Executa o teste de Johansen e retorna estatísticas de rastro e máximo autovalor."""
    try:
        data = pd.concat([y, x], axis=1)
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        
        # Retorna estatística de rastro e máximo autovalor para r=0
        return result.lr1[0], result.lr2[0]
    except Exception as e:
        logging.error(f"Erro no teste Johansen: {e}")
        return 0.0, 0.0
        
def is_integrated_of_order_one(series: pd.Series) -> bool:
    """Check if a series is I(1) using ADF test."""
    try:
        # Test for stationarity (I(0))
        adf_result_level = adfuller(series.dropna(), maxlag=10)
        p_value_level = adf_result_level[1]
        
        # If level is stationary, it's not I(1)
        if p_value_level < ADF_PVALUE_THRESHOLD:
            return False
        
        # Test first difference for stationarity
        diff_series = series.diff().dropna()
        adf_result_diff = adfuller(diff_series, maxlag=10)
        p_value_diff = adf_result_diff[1]
        
        # If level is non-stationary but difference is stationary, it's I(1)
        return p_value_diff < ADF_PVALUE_THRESHOLD
    except Exception as e:
        logging.error(f"Error in ADF testing: {e}")
        return False

def test_pair(asset1: str, asset2: str, data: pd.DataFrame, lookback: int) -> Optional[Tuple]:
    """Testa um par para cointegração com critérios mais flexíveis."""
    try:
        y = data[asset1][-lookback:].reset_index(drop=True)
        x = data[asset2][-lookback:].reset_index(drop=True)
        
        if y.isna().any() or x.isna().any() or len(y) < lookback/2 or len(x) < lookback/2:
            return None
        
        # 1. Check I(1) condition
        if not (is_integrated_of_order_one(y) and is_integrated_of_order_one(x)):
            return None
        
        # 2. Engle-Granger test with more flexible p-value
        eg_pvalue, eg_stat = engle_granger_test(y, x)
        
        # 3. Johansen test with trace statistic
        johansen_trace, _ = johansen_test(y, x)
        
        # More flexible criteria
        if eg_pvalue < ADF_PVALUE_THRESHOLD and johansen_trace > 0:
            spread, _ = compute_spread((asset1, asset2), data, lookback)
            if spread.std() >= MIN_SPREAD_STD:  # Ensure minimum volatility
                return (asset1, asset2, eg_pvalue, eg_stat, johansen_trace, spread.std())
        
        return None
    except Exception as e:
        logging.debug(f"Error testing pair ({asset1}, {asset2}): {str(e)}")
        return None

def select_cointegrated_pairs(data: pd.DataFrame, lookback: int, valid_tickers: List[str], 
                              pair_cache: Dict, use_cache: bool = True) -> List[Tuple]:
    """Seleciona pares cointegrados com cache persistente, testando ambas as ordens."""
    cache_key = (lookback, tuple(sorted(valid_tickers)))
    if use_cache and cache_key in pair_cache:
        cached_pairs = pair_cache[cache_key]
        if len(cached_pairs) > 0 and len(cached_pairs[0]) == 6:  # Verificar estrutura
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
    
    pair_combinations = [(assets[i], assets[j]) for i in range(len(assets)) for j in range(i+1, len(assets))]
    
    def test_both_orders(asset1, asset2):
        result1 = test_pair(asset1, asset2, data, lookback)
        result2 = test_pair(asset2, asset1, data, lookback)
        
        # Se os dois forem válidos, escolher o de menor p-valor
        if result1 and result2:
            return result1 if result1[2] <= result2[2] else result2
        # Se apenas um for válido, retornar esse
        return result1 or result2

    results = Parallel(n_jobs=-1)(
        delayed(test_both_orders)(asset1, asset2) for asset1, asset2 in pair_combinations
    )
    
    pairs = [r for r in results if r is not None]
    pair_cache[cache_key] = pairs
    
    logging.info(f"Tempo de seleção de pares para lookback {lookback}: {time.time() - start_time:.2f} segundos")
    logging.info(f"Total pares testados (duas ordens): {len(pair_combinations) * 2}, Pares cointegrados encontrados: {len(pairs)}")
    return pairs
    
def compute_spread(pair: Tuple[str, str], data: pd.DataFrame, lookback: int) -> Tuple[pd.Series, float]:
    """Calcula o spread para um par cointegrado."""
    asset1, asset2 = pair[:2]
    y = data[asset1][-lookback:].reset_index(drop=True)
    x = data[asset2][-lookback:].reset_index(drop=True)
    
    if y.isna().any() or x.isna().any():
        raise ValueError(f"Dados insuficientes para o par ({asset1}, {asset2})")
    
    model = OLS(y, sm.add_constant(x)).fit()
    hedge_ratio = model.params.iloc[1]
    intercept = model.params.iloc[0]
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
              portfolio_value: float, position_state: Dict, trade_log: List, timestamp: datetime) -> Tuple[float, float, int]:
    """Executa a lógica de trading para um par com controle de holding."""
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
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}
        
        current_z = z_score.iloc[-1]
        current_position = position_state[pair_key]['position']
        hold_days = position_state[pair_key]['hold_days']
        entry_value = position_state[pair_key]['entry_value']
        
        if current_position != 0:
            position_state[pair_key]['hold_days'] += 1
        
        # Calcular retorno atual se estivermos em uma posição
        if current_position > 0:  # Long position
            current_return = (spread.iloc[-1] - spread.iloc[-2]) * position_size / portfolio_value
        elif current_position < 0:  # Short position
            current_return = -(spread.iloc[-1] - spread.iloc[-2]) * position_size / portfolio_value
        else:
            current_return = 0
        
        # Verificar stop loss
        if current_position != 0 and (current_return / (position_size/portfolio_value)) <= -STOP_LOSS:
            returns = -STOP_LOSS * (position_size/portfolio_value) - TRANSACTION_COST * 2
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}
            trade_count = 1
            trade_log.append([timestamp, pair[0], pair[1], current_z, returns, 'stop_loss'])
            return returns, 0, trade_count
        
        # Verificar max hold days
        if hold_days >= MAX_HOLD_DAYS and current_position != 0:
            returns = current_return - TRANSACTION_COST * 2
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}
            trade_count = 1
            trade_log.append([timestamp, pair[0], pair[1], current_z, returns, 'max_days'])
            return returns, 0, trade_count
        
        # Lógica de trading
        if current_z < -Z_ENTRY_THRESHOLD and current_position == 0:  # Entrada long
            position_state[pair_key] = {
                'position': position_size, 
                'hold_days': 0,
                'entry_value': portfolio_value
            }
            returns = 0  # Sem retorno no dia de entrada
            trade_count = 1
            trade_log.append([timestamp, pair[0], pair[1], current_z, returns, 'long'])
        elif current_z > Z_ENTRY_THRESHOLD and current_position == 0:  # Entrada short
            position_state[pair_key] = {
                'position': -position_size, 
                'hold_days': 0,
                'entry_value': portfolio_value
            }
            returns = 0  # Sem retorno no dia de entrada
            trade_count = 1
            trade_log.append([timestamp, pair[0], pair[1], current_z, returns, 'short'])
        elif current_position > 0 and current_z > Z_EXIT_THRESHOLD_LONG:  # Saída long
            returns = current_return - TRANSACTION_COST * 2
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}
            trade_count = 1
            trade_log.append([timestamp, pair[0], pair[1], current_z, returns, 'close_long'])
        elif current_position < 0 and current_z < Z_EXIT_THRESHOLD_SHORT:  # Saída short
            returns = current_return - TRANSACTION_COST * 2
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}
            trade_count = 1
            trade_log.append([timestamp, pair[0], pair[1], current_z, returns, 'close_short'])
        else:  # Manter posição
            returns = current_return
        
        return returns, position_state[pair_key]['position'], trade_count
    except Exception as e:
        logging.error(f"Erro ao negociar par {pair}: {e}")
        return 0, position_state.get(pair_key, {'position': 0, 'hold_days': 0, 'entry_value': 0})['position'], 0
        
def backtest_strategy(pairs: List[Tuple], data: pd.DataFrame, lookback: int,
                     valid_tickers: List[str], period_idx: int, trading_start: datetime) -> Tuple[float, List, int, List]:
    """Realiza backtesting para uma combinação de parâmetros."""
    start_time = time.time()
    portfolio_value = 100000
    portfolio_returns = []
    total_trades = 0
    position_state = {}
    trade_log = []
    
    active_tickers = [t for t in valid_tickers if t in data.columns and data[t].isna().mean() < 0.20]  # Aumentado para 0.20
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
    returns_series: pd.Series,
    trade_log: List[tuple],
    rf_series: pd.Series
) -> Dict[str, float]:
    """
    Calcula métricas de performance a partir de:
      - returns_series: série de retornos diários (pd.Series, index datetime)
      - trade_log: lista de tuplas/objetos com informação de cada trade
      - rf_series: série de retornos diários risk-free (pd.Series, index datetime)
    Retorna dicionário com:
      - cumulative_return
      - annualized_return
      - sharpe_ratio
      - sortino_ratio
      - omega_ratio
      - kappa_3_ratio
      - calmar_ratio
      - sterling_ratio
      - max_drawdown
      - win_rate
      - avg_trade_return
      - num_trades
    """
    # 1) alinhar rf e calcular retornos excedentes
    rf_aligned = rf_series.reindex(returns_series.index).fillna(method='ffill').fillna(0.0)
    excess_returns = returns_series - rf_aligned

    # 2) métricas básicas
    cumulative_return = (1 + returns_series).prod() - 1
    days = returns_series.shape[0]
    years = days / 365 if days > 0 else np.nan

    # 3) Retorno anualizado (geométrico)
    if days > 0 and cumulative_return > -1 and years > 0:
        annualized_return = (1 + cumulative_return) ** (1 / years) - 1
    else:
        annualized_return = np.nan

    # 4) Sharpe Ratio
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std(ddof=1)
    sharpe_ratio = (np.sqrt(365) * mean_excess / std_excess) if std_excess > 0 else np.nan

    # 5) Sortino Ratio (downside risk)
    negative = returns_series.copy()
    negative[negative > 0] = 0
    downside_std = negative.std(ddof=1)
    sortino_ratio = (np.sqrt(365) * mean_excess / downside_std) if downside_std > 0 else np.nan

    # 6) Omega Ratio (threshold = 0)
    gains = returns_series[returns_series > 0].sum()
    losses = -returns_series[returns_series < 0].sum()
    omega_ratio = gains / losses if losses > 0 else np.inf

    # 7) Kappa-3 Ratio (penalidade cúbica de drawdown)
    kappa_3_ratio = np.nan  # placeholder; calc. customizado pode ser inserido aqui

    # 8) Máximo e drawdowns
    cum = (1 + returns_series).cumprod()
    running_max = cum.cummax()
    drawdowns = cum / running_max - 1
    max_dd = drawdowns.min()
    avg_dd = drawdowns[drawdowns < 0].mean()

    # 9) Calmar e Sterling
    calmar_ratio = (annualized_return / abs(max_dd)) if max_dd < 0 else np.inf
    sterling_ratio = (annualized_return / abs(avg_dd)) if avg_dd < 0 else np.inf

    # 10) Métricas de trades: só contas as saídas (round-trips)
    exit_types = {'close_long', 'close_short', 'stop_loss', 'max_days'}
    exit_trades = [t for t in trade_log if t[5] in exit_types]

    num_trades = len(exit_trades)
    wins = sum(1 for t in exit_trades if t[4] > 0)    # t[4] = retorno do trade na saída
    win_rate = wins / num_trades if num_trades > 0 else np.nan
    avg_trade_return = np.mean([t[4] for t in exit_trades]) if num_trades > 0 else np.nan

    return {
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'omega_ratio': omega_ratio,
        'kappa_3_ratio': kappa_3_ratio,
        'calmar_ratio': calmar_ratio,
        'sterling_ratio': sterling_ratio,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'num_trades': num_trades
    }

def identify_crisis_periods(market_index: pd.DataFrame, threshold=-0.20) -> pd.DataFrame:
    """
    Versão corrigida que:
    1. Compara corretamente preços (float vs float)
    2. Elimina comparações entre tipos diferentes
    3. Inclui validações adicionais
    """
    if isinstance(market_index, pd.DataFrame):
        if 'Close' not in market_index.columns:
            raise ValueError("DataFrame do índice de mercado deve conter coluna 'Close'")
        prices = market_index['Close'].dropna().sort_index()
    else:
        prices = market_index.dropna().sort_index()
    
    if len(prices) == 0:
        return pd.DataFrame()

    crises = []
    peak = prices.iloc[0]  # Valor float do preço
    peak_date = prices.index[0]  # Objeto Timestamp da data
    in_crisis = False

    for date, price in prices.items():
        if not isinstance(price, (float, int)):
            continue  # Ignora valores não numéricos

        if not in_crisis:
            if price > peak:  # Agora comparando float com float
                peak = price
                peak_date = date
            drawdown = (price - peak) / peak
            if drawdown <= threshold:
                in_crisis = True
                start_date = date
                trough = price
                trough_date = date
        else:
            if price < trough:
                trough = price
                trough_date = date
            recovery = (price - trough) / trough
            if recovery >= abs(threshold):
                end_date = date
                crises.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'peak_date': peak_date,
                    'trough_date': trough_date,
                    '%_decline': round((trough - peak) / peak * 100, 2),
                    'duration_days': (end_date - start_date).days + 1
                })
                in_crisis = False
                peak = price
                peak_date = date

    # Validação final
    if in_crisis:
        end_date = prices.index[-1]
        crises.append({
            'start_date': start_date,
            'end_date': end_date,
            'peak_date': peak_date,
            'trough_date': trough_date,
            '%_decline': round((trough - peak) / peak * 100, 2),
            'duration_days': (end_date - start_date).days + 1
        })

    return pd.DataFrame(crises)

def analyze_crisis_performance(returns: pd.Series, market_index: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa o desempenho da estratégia durante períodos de crise e não crise.
    Agora com contagem precisa de dias e verificação de integridade.
    """
    crises = identify_crisis_periods(market_index)
    returns = returns.sort_index()
    
    # Verifica se temos dados para todo o período
    start_date = returns.index.min()
    end_date = returns.index.max()
    total_days = (end_date - start_date).days + 1
    
    results = {
        'crisis_periods': crises,
        'crisis_stats': None,
        'non_crisis_stats': None,
        'comparison': None,
        'total_days': total_days,
        'accounted_days': 0
    }
    
    if crises.empty:
        results['non_crisis_stats'] = {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'total_return': (1 + returns).prod() - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(365),
            'days': len(returns)
        }
        results['accounted_days'] = len(returns)
        return results
    
    # Marcadores para cada dia se está em crise ou não
    crisis_mask = pd.Series(False, index=returns.index)
    for _, crisis in crises.iterrows():
        crisis_mask.loc[crisis['start_date']:crisis['end_date']] = True
    
    # Estatísticas durante crises
    crisis_returns = returns[crisis_mask]
    if not crisis_returns.empty:
        results['crisis_stats'] = {
            'mean_return': crisis_returns.mean(),
            'std_return': crisis_returns.std(),
            'total_return': (1 + crisis_returns).prod() - 1,
            'sharpe_ratio': crisis_returns.mean() / crisis_returns.std() * np.sqrt(365),
            'days': len(crisis_returns)
        }
        results['accounted_days'] += len(crisis_returns)
    
    # Estatísticas fora de crises
    non_crisis_returns = returns[~crisis_mask]
    if not non_crisis_returns.empty:
        results['non_crisis_stats'] = {
            'mean_return': non_crisis_returns.mean(),
            'std_return': non_crisis_returns.std(),
            'total_return': (1 + non_crisis_returns).prod() - 1,
            'sharpe_ratio': non_crisis_returns.mean() / non_crisis_returns.std() * np.sqrt(365),
            'days': len(non_crisis_returns)
        }
        results['accounted_days'] += len(non_crisis_returns)
    
    # Verificação de integridade
    if results['accounted_days'] != len(returns):
        logging.warning(f"Discrepância na contagem de dias: Total={len(returns)}, Contabilizados={results['accounted_days']}")
    
    # Comparação
    if results['crisis_stats'] and results['non_crisis_stats']:
        results['comparison'] = {
            'return_ratio': results['crisis_stats']['total_return'] / results['non_crisis_stats']['total_return'],
            'sharpe_ratio_diff': results['crisis_stats']['sharpe_ratio'] - results['non_crisis_stats']['sharpe_ratio'],
            'outperformance': results['crisis_stats']['total_return'] - results['non_crisis_stats']['total_return'],
            'days_diff': results['crisis_stats']['days'] - results['non_crisis_stats']['days']
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
    idx  = global_metrics.get('daily_returns_index')
    if rets is None or idx is None:
        logging.error("daily_returns ou daily_returns_index faltando em global_metrics")
        return
    if not isinstance(rets, pd.Series):
        rets = pd.Series(rets, index=idx)
    rets.index = pd.to_datetime(rets.index)
    rets = rets.sort_index().loc['2018-01-01':'2024-12-31']
    logging.info(f"Dias de dados: {len(rets)}")
    if rets.empty:
        logging.error("Nenhum dado no intervalo 01/2018–12/2024")
        return

    # 3. Calcula equity e drawdown
    df = pd.DataFrame({'returns': rets})
    df['equity']   = 100 * (1 + df['returns']).cumprod()
    df['peak']     = df['equity'].cummax()
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

    # 5. Drawdown
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

    logging.info("Plotando distribuição dos retornos com zoom na cauda")

    trade_log = global_metrics.get("trade_log", [])
    if isinstance(trade_log, list) and len(trade_log) > 0:
        trade_df = pd.DataFrame(trade_log, columns=["date", "asset1", "asset2", "z", "ret", "type"])
        exit_types = ["close_long", "close_short", "stop_loss", "max_days"]
        trade_df = trade_df[trade_df["type"].isin(exit_types)]

        if not trade_df.empty:
            trade_returns = trade_df["ret"] * 100
            minb = np.floor(trade_returns.min() * 2) / 2
            maxb = np.ceil(trade_returns.max() * 2) / 2
            bins = np.arange(minb, maxb + 0.1, 0.1)

            # Gráfico completo
            fig, ax = plt.subplots(figsize=(12, 6))
            n, bins, patches = ax.hist(trade_returns, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Retorno por Trade (%)')
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.arange(minb, maxb + 0.5, 0.5))  # Ajuste aqui
            fig.autofmt_xdate()
            save_plot(fig, 'return_distribution_full.png')

            # Gráfico com zoom na cauda (ex: retornos > 1%)
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
                ax.set_xticks(np.arange(minz, maxz + 0.5, 0.5))  # Ajuste aqui

                # Adiciona valores nas barras
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


    # 7. Sharpe Ratio por Período
    logging.info("Plotando Sharpe por período")
    sharpe_global = global_metrics.get('sharpe_ratio', np.nan)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(period_metrics['period'], period_metrics['sharpe_ratio'],
           width=0.8, edgecolor='black', alpha=0.7)
    ax.axhline(sharpe_global, color='black', linestyle='--',
               label='Sharpe Global')
    ax.set_title('Sharpe Ratio por Período')
    ax.set_xlabel('Período')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(period_metrics['period'])
    ax.legend()
    fig.autofmt_xdate()
    save_plot(fig, 'sharpe_per_period.png')

    logging.info("plot_results concluído.")
    
def check_partial_results() -> None:
    """Verifica quais períodos já foram processados."""
    if not os.path.exists(METRICS_CACHE_FILE):
        print("Nenhum resultado em cache encontrado.")
        return

    with open(METRICS_CACHE_FILE, 'rb') as f:
        metrics_cache = pickle.load(f)
    
    processed_periods = sorted([int(k.split('_')[1]) for k in metrics_cache.keys()])
    print(f"Períodos processados: {processed_periods}")
    print(f"Total: {len(processed_periods)} períodos concluídos")
    
    if processed_periods:
        sharpe_list = [metrics_cache[f"period_{p}"]['sharpe'] for p in processed_periods]
        print(f"\nSharpe médio: {np.mean(sharpe_list):.2f}")
        print(f"Melhor Sharpe: {np.max(sharpe_list):.2f} (período {processed_periods[np.argmax(sharpe_list)]})")
        
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

def optimize_strategy(
    periods: pd.DataFrame,
    pt_data: pd.DataFrame,
    rt_data: pd.DataFrame,
    semester_tickers: Dict[str, List[str]],
    rf: pd.Series,
    market_index: pd.DataFrame,  # Adicionado
    use_cache: bool = True,
    force_reprocess: Optional[List[int]] = None
) -> Tuple[Dict[int, dict], Dict[str, Any]]:
    """Otimiza parâmetros para cada período com suporte a cache e risk-free."""
    start_time = time.time()
    
    # ----------------------------
    # 1) Carregamento de cache
    # ----------------------------
    if use_cache:
        if not (os.path.exists(PAIRS_CACHE_FILE) and os.path.exists(METRICS_CACHE_FILE)):
            logging.warning("Cache não encontrado, processando do zero")
            use_cache = False
        else:
            logging.info("Cache disponível, carregando...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    pair_cache = {}
    if use_cache and os.path.exists(PAIRS_CACHE_FILE):
        try:
            with open(PAIRS_CACHE_FILE, 'rb') as f:
                pair_cache = pickle.load(f)
            logging.info(f"Cache de pares: {len(pair_cache)} entradas")
        except Exception as e:
            logging.error(f"Falha ao ler cache de pares: {e}")
    metrics_cache = {}
    if use_cache and os.path.exists(METRICS_CACHE_FILE):
        try:
            with open(METRICS_CACHE_FILE, 'rb') as f:
                metrics_cache = pickle.load(f)
            logging.info(f"Cache de métricas: {len(metrics_cache)} períodos")
        except Exception as e:
            logging.error(f"Falha ao ler cache de métricas: {e}")
    
    # Se pediu reprocessar alguns períodos, limpa do cache
    if force_reprocess:
        if isinstance(force_reprocess, int):
            force_reprocess = [force_reprocess]
        for p in force_reprocess:
            key = f"period_{p}"
            if key in metrics_cache:
                metrics_cache.pop(key)
                logging.info(f"Cache limpo para período {p}")
        with open(METRICS_CACHE_FILE, 'wb') as f:
            pickle.dump(metrics_cache, f)
    
    # ----------------------------
    # 2) Geração de janelas
    # ----------------------------
    windows = generate_rolling_windows(
        pd.to_datetime("2018-01-01"), pd.to_datetime("2024-12-31")
    )
    
    all_returns    = pd.Series(dtype=float)
    all_trade_log  = []
    results        = []
    processed      = 0
    
    # ----------------------------
    # 3) Loop principal
    # ----------------------------
    for idx, window in enumerate(tqdm(windows, desc="Períodos")):
        cache_key = f"period_{idx}"
        
        # 3.1) Se estiver no cache e não for reprocessar, reutiliza
        if use_cache and cache_key in metrics_cache and (not force_reprocess or idx not in force_reprocess):
            res = metrics_cache[cache_key]
            results.append(res)
            # reconstrói série de retornos
            serie = pd.Series(res['returns'], 
                              index=pd.date_range(res['trading_start'],
                                                  periods=res['trading_length'], freq='D'))
            all_returns = pd.concat([all_returns, serie])
            all_trade_log.extend(res['trade_log'])
            processed += 1
            continue
        
        # 3.2) Seleção de tickers do semestre
        mid = window['formation_start'] + timedelta(days=180)
        sem = f"{mid.year}-{'Jan-Jun' if mid.month<=6 else 'Jul-Dec'}"
        valid = semester_tickers.get(sem, [])
        if len(valid) < MIN_TICKERS:
            logging.warning(f"{idx}: semestral <{MIN_TICKERS} tickers, pulando")
            continue
        
        # 3.3) Dados de formação e trading
        form = pt_data[(pt_data['timestamp']>=window['formation_start']) &
                       (pt_data['timestamp']<=window['formation_end'])]
        if len(form)<max(LOOKBACKS):
            logging.warning(f"{idx}: dados insuficientes formação")
            continue
        
        trade = pt_data[(pt_data['timestamp']>=window['trading_start']) &
                        (pt_data['timestamp']<=window['trading_end'])]
        if len(trade)<10:
            logging.warning(f"{idx}: dados insuficientes trading")
            continue
        
        # 3.4) Seleção de pares
        pairs = []
        for lb in LOOKBACKS:
            pairs += select_cointegrated_pairs(form, lb, valid, pair_cache, use_cache)
        pairs = list(set(pairs))
        if len(pairs)<MIN_PAIRS:
            logging.warning(f"{idx}: <{MIN_PAIRS} pares, pulando")
            continue
        
        # 3.5) Backtest e escolha do melhor lookback
        best_sh, best_res = -np.inf, None
        for lb in LOOKBACKS:
            sh, rets, nt, log = backtest_strategy(
                pairs, trade, lb, valid, idx, window['trading_start']
            )
            if sh>best_sh:
                best_sh = sh
                best_res = {
                    'period': idx,
                    'lookback': lb,
                    'sharpe': sh,
                    'returns': rets,
                    'trade_log': log,
                    'metrics': calculate_metrics(
                        pd.Series(rets,
                                  index=pd.date_range(window['trading_start'],
                                                      periods=len(rets), freq='D')),
                        log, rf
                    ),
                    'formation_start': window['formation_start'],
                    'formation_end':   window['formation_end'],
                    'trading_start':   window['trading_start'],
                    'trading_end':     window['trading_end'],
                    'trading_length':  len(rets)
                }
        if best_res is None:
            continue
        
        # 3.6) Armazena resultados e cache
        results.append(best_res)
        metrics_cache[cache_key] = best_res
        all_returns = pd.concat([
            all_returns,
            pd.Series(best_res['returns'],
                      index=pd.date_range(window['trading_start'],
                                          periods=best_res['trading_length'], freq='D'))
        ])
        all_trade_log.extend(best_res['trade_log'])
        processed += 1
        
        if idx%5==0 or idx==len(windows)-1:
            with open(METRICS_CACHE_FILE, 'wb') as f: pickle.dump(metrics_cache, f)
            with open(PAIRS_CACHE_FILE,   'wb') as f: pickle.dump(pair_cache,   f)
    
    # ----------------------------
    # 4) Consolida resultados
    # ----------------------------
    best_results = {r['period']: r for r in results}
    
    # 5) Métricas globais
    if not all_returns.empty:
        gm = calculate_metrics(all_returns, all_trade_log, rf)
        crisis_analysis = analyze_crisis_performance(all_returns, market_index)  # Nova análise
        gm.update({
            'daily_returns': all_returns,
            'cumulative_returns': (1 + all_returns).cumprod() - 1,
            'drawdowns': ((1 + all_returns).cumprod() / (1 + all_returns).cumprod().cummax() - 1),
            'crisis_periods': crisis_analysis['crisis_periods'],
            'crisis_stats': crisis_analysis['crisis_stats'],
            'non_crisis_stats': crisis_analysis['non_crisis_stats'],
            'crisis_comparison': crisis_analysis['comparison'],
            'trade_log': all_trade_log  # ✅ Adicionado para permitir geração do gráfico de distribuição dos trades
        })
        global_metrics = gm
    else:
        global_metrics = {
            'daily_returns': pd.Series(dtype=float),
            'cumulative_returns': pd.Series(dtype=float),
            'drawdowns': pd.Series(dtype=float),
            'crisis_periods': pd.DataFrame(),
            'trade_log': []  # ✅ Adicionado para manter consistência
        }

    logging.info(f"Otimize completo em {time.time() - start_time:.2f}s "
                 f"({processed}/{len(windows)} períodos)")
    return best_results, global_metrics

def main(use_cache: bool = True, force_reprocess: Optional[List[int]] = None) -> None:
    """Função principal com controle de cache."""
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
        # 4) Executa otimização, incluindo market_index
        best_results, global_metrics = optimize_strategy(
            periods, pt_data, rt_data, semester_tickers,
            rf_series, market_index,
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

        # 6) Analisa desempenho em crises usando o mercado
        returns_series = pd.Series(
            global_metrics['daily_returns'],
            index=global_metrics['daily_returns_index']
        )
        crisis_res = analyze_crisis_performance(returns_series, market_index)
        global_metrics.update(crisis_res)

        # 7) Log de resultados de crise
        cs = global_metrics.get('crisis_stats', {})
        ncs = global_metrics.get('non_crisis_stats', {})
        comp = global_metrics.get('comparison', {})
        if cs:
            logging.info("\nDesempenho Durante Crises:")
            logging.info(f"Dias em crise           : {cs.get('days', 0)}")
            logging.info(f"Retorno total crise     : {cs.get('total_return', 0)*100:.2f}%")
            logging.info(f"Sharpe Ratio crise     : {cs.get('sharpe_ratio', 0):.2f}")
        if ncs:
            logging.info("\nDesempenho Fora de Crises:")
            logging.info(f"Dias fora de crise      : {ncs.get('days', 0)}")
            logging.info(f"Retorno total fora      : {ncs.get('total_return', 0)*100:.2f}%")
            logging.info(f"Sharpe Ratio fora      : {ncs.get('sharpe_ratio', 0):.2f}")
        if comp:
            logging.info("\nComparação:")
            logging.info(f"Retorno relativo (c/n) : {comp.get('return_ratio', 0):.2f}")
            logging.info(f"Diferença Sharpe       : {comp.get('sharpe_diff', 0):.2f}")
            logging.info(f"Outperformance absoluta: {comp.get('absolute_outperf', 0)*100:.2f}%")

        # 8) Exibe métricas globais
        logging.info("\nMétricas Globais:")
        for key, label, fmt in [
            ('cumulative_return', 'Retorno Cumulativo     ', '{:.2f}%'),
            ('annualized_return', 'Retorno Anualizado     ', '{:.2f}%'),
            ('sharpe_ratio', 'Sharpe Ratio           ', '{:.4f}'),
            ('sortino_ratio', 'Sortino Ratio          ', '{:.4f}'),
            ('omega_ratio', 'Omega Ratio            ', '{:.4f}'),
            ('kappa_3_ratio', 'Kappa 3 Ratio          ', '{:.4f}'),
            ('calmar_ratio', 'Calmar Ratio           ', '{:.4f}'),
            ('sterling_ratio', 'Sterling Ratio         ', '{:.4f}'),
            ('max_drawdown', 'Drawdown Máximo        ', '{:.2f}%'),
            ('win_rate', 'Taxa de Vitórias       ', '{:.2f}%'),
            ('avg_trade_return', 'Retorno Médio por Trade', '{:.4f}%'),
            ('num_trades', 'Total de Trades        ', '{}')
        ]:
            val = global_metrics.get(key, 0)
            if key in ['cumulative_return', 'annualized_return', 'max_drawdown', 'win_rate', 'avg_trade_return']:
                val *= 100
            logging.info(f"{label}: {fmt.format(val)}")

        # 9) Exibe períodos de crise
        cp = global_metrics.get('crisis_periods', pd.DataFrame())
        if not cp.empty:
            logging.info("\nPeríodos de Crise Identificados:")
            for rec in cp.to_dict('records'):
                logging.info(
                    f"{rec['start_date'].date()} a {rec['end_date'].date()} - "
                    f"Queda: {rec['%_decline']}% | Duração: {rec['duration_days']} dias"
                )

        # 10) Exibe métricas por período (formato único)
        logging.info("\nMétricas por Período:")
        for idx, res in best_results.items():
            msg = (
                f"Período {idx}: Lookback={res['lookback']}, "
                f"Sharpe={res['sharpe']:.4f}, "
                f"Retorno={res['metrics']['cumulative_return']*100:.2f}%, "
                f"Trades={res['metrics']['num_trades']}, "
                f"Win Rate={res['metrics']['win_rate']:.4f}"
            )
            logging.info(msg)

        logging.info(f"\nResultados salvos em {RESULTS_DIR}")
        logging.info(f"Tempo total de execução: {time.time()-start_time:.2f}s")

        # 11) Geração de gráficos contínuos
        try:
            period_metrics = pd.DataFrame([{
                'period': idx,
                'lookback': res['lookback'],  
                'sharpe_ratio': res['sharpe'],
                'cumulative_return': res['metrics']['cumulative_return']
            } for idx, res in best_results.items()]).sort_values('period')

            plot_results(period_metrics, global_metrics, ANALYSIS_DIR)
            logging.info(f"Gráficos gerados em {ANALYSIS_DIR}")
        except Exception as e:
            logging.error(f"Falha ao gerar gráficos: {e}")

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
    main(use_cache=True)
    
    # After first run, you can use cache
    # main(use_cache=True)
