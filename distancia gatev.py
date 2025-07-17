import pandas as pd
import numpy as np
import os
import time
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import shutil
import logging
import traceback
import math
from typing import List, Optional, Dict, Tuple, Any
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Diretórios
BASE_DIR = '/home/guest/project'
DATA_DIR = os.path.join(BASE_DIR, 'cointegration_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'cointegration_results', 'dist')
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
Z_ENTRY_THRESHOLD = 1.5  # 2 standard deviations for entry (as per paper)
Z_EXIT_THRESHOLD_LONG = 0.75
Z_EXIT_THRESHOLD_SHORT = 0.75
MAX_HOLD_DAYS = 50  # Máximo de dias para manter uma posição
LOOKBACKS = [90]  # 12 months formation period (252 trading days)
ADF_PVALUE_THRESHOLD = 0.10
MAX_TICKERS = 100  # Limite de tickers por período
MIN_TICKERS = 20  # Mínimo de tickers
MIN_PAIRS = 10  # Mínimo de pares cointegrados
MAX_PAIRS = 10  # Máximo de pares a serem negociados (as per paper)
TRADING_DAYS = 183  # Período de trading (6 meses)
RISK_FREE_RATE = 0.02  # Taxa livre de risco (2%)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame, pd.Series]:
    """
    Carrega:
     - Pt_cointegration.csv  → pt_data (preços)
     - Rt_cointegration.csv  → rt_data (retornos)
     - Periods.csv           → periods
     - ticker2.csv           → semester_tickers
     - Rf.csv                → rf_data (retorno diário do T-Bill)
     - market_index.csv      → market_index (preços do índice de mercado)
    """
    start_time = time.time()
    # Diretórios
    BASE_DIR = '/home/guest/project'
    DATA_DIR = os.path.join(BASE_DIR, 'cointegration_data')

    # 1) Verifica existência de todos os arquivos
    required = [
        ("Pt_cointegration.csv", "Pt_cointegration.csv"),
        ("Rt_cointegration.csv", "Rt_cointegration.csv"),
        ("Periods.csv", "Periods.csv"),
        ("ticker2.csv", "ticker2.csv"),
        ("Rf.csv", "Rf.csv"),
        ("market_index.csv", "market_index.csv"),
    ]
    for fname, desc in required:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # 2) Carrega Pt, Rt, Periods, ticker2
    pt_data = pd.read_csv(os.path.join(DATA_DIR, "Pt_cointegration.csv"), low_memory=False)
    rt_data = pd.read_csv(os.path.join(DATA_DIR, "Rt_cointegration.csv"), low_memory=False)
    periods = pd.read_csv(os.path.join(DATA_DIR, "Periods.csv"), header=None)
    tickers = pd.read_csv(os.path.join(DATA_DIR, "ticker2.csv"))

    # 3) Processa timestamps e NaNs em pt_data
    pt_data['timestamp'] = pd.to_datetime(pt_data['timestamp'])
    for col in pt_data.columns:
        if col == 'timestamp': 
            continue
        pt_data[col] = pd.to_numeric(pt_data[col], errors='coerce')
        nan_ratio = pt_data[col].isna().mean()
        if nan_ratio < 0.20:
            pt_data[col] = pt_data[col].interpolate().ffill().bfill()
        else:
            logging.warning(f"Ticker {col} removido: NaN ratio={nan_ratio:.4f}")
            pt_data.drop(columns=[col], inplace=True)

    # 4) Processa timestamps e converte rt_data para numérico
    rt_data['timestamp'] = pd.to_datetime(rt_data['timestamp'])
    for col in rt_data.columns:
        if col == 'timestamp': 
            continue
        rt_data[col] = pd.to_numeric(rt_data[col], errors='coerce')

    # 5) Monta semester_tickers por semestre
    semester_tickers = {}
    semesters = [(y, 1) for y in range(2018, 2025)] + [(y, 2) for y in range(2018, 2025)]
    for idx, col in enumerate(tickers.columns):
        if idx >= len(semesters): 
            break
        year, half = semesters[idx]
        key = f"{year}-{'Jan-Jun' if half == 1 else 'Jul-Dec'}"
        start = pd.to_datetime(f"{year}-{'01-01' if half == 1 else '07-01'}")
        end = pd.to_datetime(f"{year}-{'06-30' if half == 1 else '12-31'}")
        subset = pt_data[(pt_data['timestamp'] >= start) & (pt_data['timestamp'] <= end)]
        valid = []
        for t in tickers[col].dropna().unique():
            if t in pt_data.columns:
                ratio = subset[t].isna().mean()
                if ratio < 0.20:
                    valid.append(t)
                else:
                    logging.warning(f"{t} ignorado em {key}: NaN ratio={ratio:.4f}")
        semester_tickers[key] = valid[:MAX_TICKERS]
        logging.info(f"Semestre {key}: {len(semester_tickers[key])} tickers válidos")

    # 6) Carrega índice de mercado
    market_index = pd.read_csv(MARKET_INDEX_FILE, parse_dates=['Date'])
    market_index.set_index('Date', inplace=True)
    market_index = market_index.sort_index()
    if 'Close' not in market_index.columns:
        logging.error("market_index.csv deve conter a coluna 'Close'")
        raise ValueError("market_index.csv deve conter a coluna 'Close'")
    market_index = market_index['Close']
    # Valida preços inválidos
    if (market_index <= 0).any():
        logging.warning(f"market_index contém {sum(market_index <= 0)} valores zero ou negativos. Substituindo por NaN e interpolando.")
        market_index = market_index.where(market_index > 0).interpolate().ffill().bfill()
    if market_index.isna().any():
        logging.warning("market_index contém NaNs após interpolação. Usando valores padrão.")
        market_index = market_index.fillna(method='ffill').fillna(method='bfill')

    # 7) Carrega Rf.csv dinamicamente, renomeia a coluna de taxa para 'rf'
    rf_path = os.path.join(DATA_DIR, "Rf.csv")
    rf_data = pd.read_csv(rf_path)
    date_col = rf_data.columns[0]
    rf_data[date_col] = pd.to_datetime(rf_data[date_col])
    rf_data.set_index(date_col, inplace=True)
    rate_cols = [c for c in rf_data.columns if c.lower() not in ['date', 'dt']]
    if not rate_cols:
        raise KeyError("Não foi possível identificar coluna de retornos em Rf.csv")
    rf_data.rename(columns={rate_cols[0]: 'rf'}, inplace=True)
    if rf_data['rf'].max() > 1:
        rf_data['rf'] /= 100.0

    logging.info(f"Tempo de carregamento: {time.time() - start_time:.2f}s")
    return pt_data, rt_data, periods, semester_tickers, rf_data, market_index

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

def select_pairs_by_distance(formation_data: pd.DataFrame, valid_tickers: List[str]) -> List[Tuple]:
    """Seleciona pares com base no método de distância (SSD) conforme o artigo."""
    start_time = time.time()
    assets = [t for t in valid_tickers if t in formation_data.columns and formation_data[t].isna().mean() < 0.20]
    
    if len(assets) < MIN_TICKERS:
        logging.warning(f"Ignorando período: apenas {len(assets)} tickers (< {MIN_TICKERS})")
        return []
    
    # 1. Gerar todas as combinações possíveis de pares
    pair_combinations = [(assets[i], assets[j]) for i in range(len(assets)) for j in range(i+1, len(assets))]
    
    # 2. Calcular SSD para cada par durante o período de formação (12 meses)
    ssd_results = Parallel(n_jobs=-1)(
        delayed(calculate_ssd)(pair, formation_data.iloc[-365:]) for pair in pair_combinations
    )
    
    # 3. Criar dataframe com pares e SSDs
    pairs_df = pd.DataFrame({
        'pair': pair_combinations,
        'ssd': ssd_results
    })

    # 4. Ordenar por SSD (menor primeiro) e selecionar os top 20 pares
    pairs_df = pairs_df.sort_values('ssd').reset_index(drop=True).head(MAX_PAIRS)
    
    # 5. Para cada par selecionado, calcular o spread e seu desvio padrão
    selected_pairs = []
    for _, row in pairs_df.iterrows():
        asset1, asset2 = row['pair']
        prices1 = formation_data[asset1].iloc[-365:].dropna()
        prices2 = formation_data[asset2].iloc[-365:].dropna()
        
        # Normalizar preços
        norm_prices1 = prices1 / prices1.iloc[0]
        norm_prices2 = prices2 / prices2.iloc[0]
        
        # Calcular spread e desvio padrão
        spread = norm_prices1 - norm_prices2
        spread_std = spread.std()
        
        # Adicionar à lista de pares selecionados
        selected_pairs.append((asset1, asset2, 0, 0, 1.0, spread_std))  # hedge_ratio = 1.0 para método de distância
    
    logging.info(f"Selecionados {len(selected_pairs)} pares pelo método de distância em {time.time() - start_time:.2f} segundos")
    return selected_pairs

def compute_spread(pair: Tuple[str, str], data: pd.DataFrame, lookback: int) -> Tuple[pd.Series, float]:
    asset1, asset2, _, _, hedge_ratio, _ = pair[:6]
    y = data[asset1][-lookback:].reset_index(drop=True)
    x = data[asset2][-lookback:].reset_index(drop=True)
    
    if y.isna().any() or x.isna().any():
        logging.warning(f"Missing data in pair {asset1}-{asset2}")
        raise ValueError(f"Dados insuficientes para o par ({asset1}, {asset2})")
    
    spread = y - hedge_ratio * x
    logging.debug(f"Spread stats for {asset1}-{asset2}: Mean={spread.mean():.4f}, Std={spread.std():.4f}")
    return spread, hedge_ratio

def plot_sample_spread(pair, data, lookback):
    asset1, asset2 = pair[0], pair[1]
    spread = data[asset1][-lookback:] - data[asset2][-lookback:]
    
    plt.figure(figsize=(12,6))
    spread.plot(title=f"Spread {asset1}-{asset2}")
    plt.axhline(spread.mean(), color='black')
    plt.axhline(spread.mean() + spread.std(), color='red')
    plt.axhline(spread.mean() - spread.std(), color='green')
    plt.show()

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
        # Calcular spread e z-score
        spread, hedge_ratio = compute_spread(pair, data, lookback)
        z_score = (spread - spread.mean()) / spread.std()
        spread_std = spread.std()
        current_z = z_score.iloc[-1]

        if spread_std < MIN_SPREAD_STD:
            return 0, position_state.get(pair_key, {'position': 0, 'hold_days': 0})['position'], 0

        position_size = calculate_position_size(spread, portfolio_value)
        returns = 0
        trade_count = 0

        if pair_key not in position_state:
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}

        current_position = position_state[pair_key]['position']
        hold_days = position_state[pair_key]['hold_days']

        if current_position != 0:
            position_state[pair_key]['hold_days'] += 1

        # Calcular retorno atual se estiver posicionado
        if current_position > 0:  # Posição Long
            current_return = (spread.iloc[-1] - spread.iloc[-2]) * position_size / portfolio_value
            position_type = "long"
        elif current_position < 0:  # Posição Short
            current_return = -(spread.iloc[-1] - spread.iloc[-2]) * position_size / portfolio_value
            position_type = "short"
        else:
            current_return = 0
            position_type = "none"

        # Verifica Stop Loss
        if current_position != 0:
            # Certifique-se de que 'entry_spread' foi salvo ao abrir a posição!
            entry_spread = position_state[pair_key].get('entry_spread', None)
            if entry_spread is not None and abs(entry_spread) > 1e-8:
                # Calcula P&L acumulado da posição em relação ao valor do spread de entrada
                if current_position > 0:  # Long
                    pnl_pct = (spread.iloc[-1] - entry_spread) / abs(entry_spread)
                else:  # Short
                    pnl_pct = (entry_spread - spread.iloc[-1]) / abs(entry_spread)

                # Checa stop-loss
                if pnl_pct <= -STOP_LOSS:
                    returns = -STOP_LOSS * (position_size / portfolio_value) - TRANSACTION_COST * 2
                    position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0, 'entry_spread': 0}
                    trade_count = 1
                    trade_log.append({
                        'timestamp': timestamp,
                        'asset1': pair[0],
                        'asset2': pair[1],
                        'z_score': current_z,
                        'return': returns,
                        'trade_type': f'stop_loss_{"long" if current_position > 0 else "short"}'
                    })
                    return returns, 0, trade_count

        # Verifica Max Hold Days
        if hold_days >= MAX_HOLD_DAYS and current_position != 0:
            returns = current_return - TRANSACTION_COST * 2
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}
            trade_count = 1
            trade_log.append({
                'timestamp': timestamp,
                'asset1': pair[0],
                'asset2': pair[1],
                'z_score': current_z,
                'return': returns,
                'trade_type': f'max_days_{position_type}'
            })
            return returns, 0, trade_count

        # Condições de entrada
        if current_position == 0 and abs(current_z) > Z_ENTRY_THRESHOLD:
            if current_z < -Z_ENTRY_THRESHOLD:  # Entrada Long
                position_state[pair_key] = {
                    'position': position_size,
                    'hold_days': 0,
                    'entry_value': portfolio_value,
                    'entry_z': current_z
                }
                trade_count = 1
                trade_log.append({
                    'timestamp': timestamp,
                    'asset1': pair[0],
                    'asset2': pair[1],
                    'z_score': current_z,
                    'return': 0,
                    'trade_type': 'long_entry'
                })

            elif current_z > Z_ENTRY_THRESHOLD:  # Entrada Short
                position_state[pair_key] = {
                    'position': -position_size,
                    'hold_days': 0,
                    'entry_value': portfolio_value,
                    'entry_z': current_z
                }
                trade_count = 1
                trade_log.append({
                    'timestamp': timestamp,
                    'asset1': pair[0],
                    'asset2': pair[1],
                    'z_score': current_z,
                    'return': 0,
                    'trade_type': 'short_entry'
                })

        # Condições de saída
        elif current_position > 0 and current_z >= Z_EXIT_THRESHOLD_LONG:
            returns = current_return - TRANSACTION_COST * 2
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}
            trade_count = 1
            trade_log.append({
                'timestamp': timestamp,
                'asset1': pair[0],
                'asset2': pair[1],
                'z_score': current_z,
                'return': returns,
                'trade_type': 'long_exit'
            })

        elif current_position < 0 and current_z <= Z_EXIT_THRESHOLD_SHORT:
            returns = current_return - TRANSACTION_COST * 2
            position_state[pair_key] = {'position': 0, 'hold_days': 0, 'entry_value': 0}
            trade_count = 1
            trade_log.append({
                'timestamp': timestamp,
                'asset1': pair[0],
                'asset2': pair[1],
                'z_score': current_z,
                'return': returns,
                'trade_type': 'short_exit'
            })

        # Se mantiver posição, retorna retorno diário
        elif current_position != 0:
            returns = current_return

        return returns, position_state[pair_key]['position'], trade_count

    except Exception as e:
        logging.error(f"Error trading pair {pair_key}: {str(e)}")
        logging.error(traceback.format_exc())
        return 0, position_state.get(pair_key, {'position': 0, 'hold_days': 0, 'entry_value': 0})['position'], 0

def backtest_strategy(pairs: List[Tuple], data: pd.DataFrame, lookback: int,
                     valid_tickers: List[str], period_idx: int, trading_start: datetime) -> Tuple[float, List, int, List]:
    """Realiza backtesting para uma combinação de parâmetros."""
    start_time = time.time()
    portfolio_value = 100000
    portfolio_returns_list = []
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
        portfolio_returns_list.append(daily_returns)
        total_trades += daily_trades
    
    # Convert list to Series with proper dates
    portfolio_returns = pd.Series(
        portfolio_returns_list,
        index=trading_dates[:len(portfolio_returns_list)]
    )
    
    # Pad returns to 183 days if needed
    if len(portfolio_returns) < TRADING_DAYS:
        padded_returns = np.zeros(TRADING_DAYS)
        padded_returns[:len(portfolio_returns)] = portfolio_returns.values
        portfolio_returns = pd.Series(
            padded_returns,
            index=pd.date_range(trading_start, periods=TRADING_DAYS, freq='D')
        )
    
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(365) if np.std(portfolio_returns) > 0 else 0
    logging.info(f"Tempo de backtesting (lookback={lookback}): {time.time() - start_time:.2f} segundos, {total_trades} trades")
    
    return sharpe, portfolio_returns.tolist(), total_trades, trade_log

def calculate_metrics(returns, trade_log, rf_data):
    """
    Calcula métricas de desempenho a partir de retornos diários e log de trades fechados.
    Só considera trades fechados para as métricas de trade!
    """
    metrics = {
        'cumulative_return': 0.0,
        'annualized_return': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'omega_ratio': 0.0,
        'kappa_3_ratio': 0.0,
        'calmar_ratio': 0.0,
        'sterling_ratio': 0.0,
        'max_drawdown': 0.0,
        'avg_drawdown': 0.0,
        'num_trades': 0,
        'avg_trade_return': 0.0,
        'win_rate': 0.0
    }

    if isinstance(returns, pd.Series) and not returns.empty and isinstance(rf_data, pd.DataFrame) and not rf_data.empty:
        # Calcula retornos em excesso
        excess_ret = returns - rf_data['rf'].reindex(returns.index).ffill().fillna(0.0)
        
        # Retorno cumulativo
        metrics['cumulative_return'] = (1 + excess_ret).cumprod().iloc[-1] - 1
        
        # Retorno anualizado
        days = (returns.index[-1] - returns.index[0]).days if len(returns) > 1 else 1
        metrics['annualized_return'] = (1 + metrics['cumulative_return']) ** (252 / max(days, 1)) - 1
        
        # Desvio padrão anualizado
        std = excess_ret.std() * np.sqrt(252) if excess_ret.std() != 0 else 1e-9
        
        # Sharpe Ratio
        metrics['sharpe_ratio'] = metrics['annualized_return'] / std if std != 0 else 0.0
        
        # Sortino Ratio
        downside_std = excess_ret[excess_ret < 0].std() * np.sqrt(252) if excess_ret[excess_ret < 0].std() != 0 else 1e-9
        metrics['sortino_ratio'] = metrics['annualized_return'] / downside_std if downside_std != 0 else 0.0
        
        # Omega Ratio
        threshold = 0.0
        gains = excess_ret[excess_ret > threshold].sum()
        losses = -excess_ret[excess_ret < threshold].sum()
        metrics['omega_ratio'] = gains / losses if losses != 0 else float('inf')
        
        # Kappa 3 Ratio
        skewness = excess_ret.skew()
        metrics['kappa_3_ratio'] = metrics['annualized_return'] / abs(skewness) if skewness != 0 else 0.0
        
        # Drawdowns
        equity = (1 + excess_ret).cumprod()
        drawdowns = equity / equity.cummax() - 1
        metrics['max_drawdown'] = drawdowns.min()
        metrics['avg_drawdown'] = drawdowns[drawdowns < 0].mean() if drawdowns[drawdowns < 0].size > 0 else 0.0
        
        # Calmar Ratio
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0.0
        
        # Sterling Ratio
        metrics['sterling_ratio'] = metrics['annualized_return'] / abs(metrics['avg_drawdown']) if metrics['avg_drawdown'] != 0 else 0.0

    # ==== NOVO BLOCO PARA CONTABILIZAR APENAS TRADES FECHADOS ====
    if trade_log:
        # Considere apenas trades fechados
        exit_types = [
            "long_exit", "short_exit",
            "stop_loss_long", "stop_loss_short",
            "max_days_long", "max_days_short"
        ]
        trade_df = pd.DataFrame(trade_log)
        if 'trade_type' in trade_df.columns and 'return' in trade_df.columns:
            closed_trades = trade_df[trade_df['trade_type'].isin(exit_types)]
            trade_returns = closed_trades['return'].dropna().tolist()
        else:
            # fallback (se nomes das colunas forem diferentes, adapte aqui)
            trade_returns = []
        
        metrics['num_trades'] = len(trade_returns)
        metrics['avg_trade_return'] = np.mean(trade_returns) if trade_returns else 0.0
        metrics['win_rate'] = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0.0

    return metrics

def identify_crisis_periods(market_index: pd.DataFrame, threshold: float = -0.20) -> pd.DataFrame:
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
        logging.warning("market_index está vazio após remover NaN")
        return pd.DataFrame()

    crises = []
    peak = prices.iloc[0]  # Valor float do preço
    peak_date = prices.index[0]  # Objeto Timestamp da data
    in_crisis = False
    start_date = None
    trough = peak
    trough_date = peak_date

    # Calcula drawdown máximo para depuração
    drawdowns = (prices - prices.cummax()) / prices.cummax()
    logging.info(f"Drawdown máximo do market_index: {drawdowns.min():.2%}")

    for date, price in prices.items():
        if not isinstance(price, (float, int)):
            logging.warning(f"Valor não numérico encontrado em {date}: {price}. Ignorando.")
            continue  # Ignora valores não numéricos

        if not in_crisis:
            if price > peak:  # Agora comparando float com float
                peak = price
                peak_date = date
            drawdown = (price - peak) / peak if peak != 0 else 0
            if drawdown <= threshold:
                in_crisis = True
                start_date = date
                trough = price
                trough_date = date
        else:
            if price < trough:
                trough = price
                trough_date = date
            recovery = (price - trough) / trough if trough != 0 else float('inf')
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
            '%_decline': round((trough - peak) / peak * 100, 2) if peak != 0 else -100.0,
            'duration_days': (end_date - start_date).days + 1
        })

    crisis_df = pd.DataFrame(crises)
    if crisis_df.empty:
        logging.info(f"Nenhum período de crise identificado com threshold={threshold:.2%}")
    else:
        logging.info(f"Identificados {len(crisis_df)} períodos de crise com threshold={threshold:.2%}")
    return crisis_df

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
            'mean_return': returns.mean() if not returns.empty else 0.0,
            'std_return': returns.std() if not returns.empty else 0.0,
            'total_return': (1 + returns).prod() - 1 if not returns.empty else 0.0,
            'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(365)) if not returns.empty and returns.std() != 0 else 0.0,
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
            'sharpe_ratio': crisis_returns.mean() / crisis_returns.std() * np.sqrt(365) if crisis_returns.std() != 0 else 0.0,
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
            'sharpe_ratio': non_crisis_returns.mean() / non_crisis_returns.std() * np.sqrt(365) if non_crisis_returns.std() != 0 else 0.0,
            'days': len(non_crisis_returns)
        }
        results['accounted_days'] += len(non_crisis_returns)
    
    # Verificação de integridade
    if results['accounted_days'] != len(returns):
        logging.warning(f"Discrepância na contagem de dias: Total={len(returns)}, Contabilizados={results['accounted_days']}")
    
    # Comparação
    if results['crisis_stats'] and results['non_crisis_stats']:
        results['comparison'] = {
            'return_ratio': results['crisis_stats']['total_return'] / results['non_crisis_stats']['total_return'] if results['non_crisis_stats']['total_return'] != 0 else float('inf'),
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
    idx = global_metrics.get('daily_returns_index')
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

    # 6. Distribuição de retornos
    logging.info("Plotando distribuição dos retornos por trade encerrado")

    trade_log = global_metrics.get("trade_log", [])
    if isinstance(trade_log, list) and len(trade_log) > 0:
        trade_df = pd.DataFrame(trade_log)
        # Ajuste conforme nomes dos campos do trade_log deste código:
        # Tenta as duas formas: novo formato ou antigo
        cols = trade_df.columns
        if 'return' in cols and 'trade_type' in cols:
            # Usa somente trades efetivamente encerrados
            exit_types = ["long_exit", "short_exit", "stop_loss_long", "stop_loss_short", "max_days_long", "max_days_short"]
            trade_df = trade_df[trade_df["trade_type"].isin(exit_types)]

            if not trade_df.empty:
                trade_returns = trade_df["return"] * 100
                minb = np.floor(trade_returns.min() * 2) / 2
                maxb = np.ceil(trade_returns.max() * 2) / 2
                bins = np.arange(minb, maxb + 0.1, 0.1)

                # Gráfico completo
                fig, ax = plt.subplots(figsize=(12, 6))
                n, bins, patches = ax.hist(trade_returns, bins=bins, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Retorno por Trade (%)')
                ax.set_ylabel('Frequência')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(np.arange(minb, maxb + 0.5, 0.5))
                fig.autofmt_xdate()
                save_plot(fig, 'return_distribution_full.png')

                # Gráfico com zoom na cauda (> 1%)
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
            logging.warning("trade_log sem colunas esperadas ('return' e 'trade_type'); gráfico de zoom na cauda não gerado.")
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
    semester_tickers: Dict,
    rf_data: pd.DataFrame,
    market_index: pd.Series,
    use_cache: bool = True,
    force_reprocess: Optional[List[int]] = None
) -> Tuple[Dict, Dict]:
    """Otimiza parâmetros para cada período com suporte a cache e retorna métricas usando rf_data."""
    start_time = time.time()

    # 1) Verificação inicial do cache
    if use_cache:
        cache_available = os.path.exists(PAIRS_CACHE_FILE) and os.path.exists(METRICS_CACHE_FILE)
        if not cache_available:
            logging.warning("Cache não encontrado, processando tudo do zero")
            use_cache = False
        else:
            logging.info("Cache disponível, carregando...")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # 2) Carregar cache de pares
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

    # 3) Carregar cache de métricas
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

    # 4) Forçar reprocessamento
    if force_reprocess and metrics_cache:
        if isinstance(force_reprocess, int):
            force_reprocess = [force_reprocess]
        for period in force_reprocess:
            key = f"period_{period}"
            if key in metrics_cache:
                metrics_cache.pop(key)
                logging.info(f"Cache limpo para reprocessamento do período {period}")
            else:
                logging.warning(f"Período {period} não encontrado no cache para reprocessamento")
        with open(METRICS_CACHE_FILE, 'wb') as f:
            pickle.dump(metrics_cache, f)

    # 5) Gerar janelas de formação/trading
    data_start = pd.to_datetime("2018-01-01")
    data_end = pd.to_datetime("2024-12-31")
    windows = generate_rolling_windows(data_start, data_end)

    all_returns = pd.Series(dtype=float)
    all_trade_log = []
    results = []
    processed = 0

    # 6) Loop por período
    for period_idx, window in enumerate(tqdm(windows, desc="Processando períodos")):
        cache_key = f"period_{period_idx}"
        # 6.1) Reusar cache?
        if use_cache and cache_key in metrics_cache and (not force_reprocess or period_idx not in force_reprocess):
            logging.info(f"Usando cache para período {period_idx}")
            res = metrics_cache[cache_key]
            results.append(res)
            # Reconstrói all_returns e all_trade_log
            rts = res['returns'][:res['trading_length']]
            idxs = pd.date_range(start=res['trading_start'], periods=len(rts), freq='D')
            if len(rts):
                series = pd.Series(rts, index=idxs)
                all_returns = series.combine_first(all_returns) if not all_returns.empty else series
            all_trade_log.extend(res['trade_log'])
            processed += 1
            continue

        # 6.2) Seleção de tickers válidos
        midpoint = window['formation_start'] + timedelta(days=180)
        sem_key = f"{midpoint.year}-{'Jan-Jun' if midpoint.month<=6 else 'Jul-Dec'}"
        valid = semester_tickers.get(sem_key, [])
        if len(valid) < MIN_TICKERS:
            logging.warning(f"Período {period_idx}: apenas {len(valid)} tickers (< {MIN_TICKERS})")
            continue

        logging.info(
            f"Processando período {period_idx}: Formação {window['formation_start'].date()} a "
            f"{window['formation_end'].date()}, Trading {window['trading_start'].date()} a {window['trading_end'].date()}"
        )

        # 6.3) Dados de formação
        formation = pt_data[
            (pt_data['timestamp'] >= window['formation_start']) &
            (pt_data['timestamp'] <= window['formation_end'])
        ]
        if len(formation) < max(LOOKBACKS):
            logging.warning(f"Período {period_idx} ignorado: formação insuficiente ({len(formation)} linhas)")
            continue

        # 6.4) Seleção de pares (já limitada por MAX_PAIRS internamente)
        pairs = select_pairs_by_distance(formation, valid)
        if len(pairs) < MIN_PAIRS:
            logging.warning(f"Período {period_idx} ignorado: {len(pairs)} pares (< {MIN_PAIRS})")
            continue

        # Salvar pares deste período
        pd.DataFrame(pairs, columns=[
            'asset1','asset2','eg_pvalue','eg_stat','hedge_ratio','spread_std'
        ]).to_csv(
            os.path.join(RESULTS_DIR, f"pairs_period_{period_idx}.csv"),
            index=False
        )
        logging.info(f"Período {period_idx}: {len(pairs)} pares selecionados")

        # 6.5) Dados de trading
        trading = pt_data[
            (pt_data['timestamp'] >= window['trading_start']) &
            (pt_data['timestamp'] <= window['trading_end'])
        ]
        combined = pd.concat([formation, trading]).reset_index(drop=True)
        if len(trading) < 10:
            logging.warning(f"Período {period_idx} ignorado: trading insuficiente ({len(trading)} linhas)")
            continue

        # 6.6) Backtest para cada lookback e escolhe o melhor
        best_sharpe = -np.inf
        best_res = None
        all_period_returns = []
        total_trades = 0

        for lookback in LOOKBACKS:
            sharpe, rets, trades, log_ = backtest_strategy(
                pairs, combined, lookback, valid, period_idx, window['trading_start']
            )
            total_trades += trades
            all_period_returns.extend(rets)
            if sharpe > best_sharpe:
                metrics = calculate_metrics(pd.Series(rets, index=pd.date_range(window['trading_start'], periods=len(rets), freq='D')), log_, rf_data)
                best_sharpe = sharpe
                best_res = {
                    'period': period_idx,
                    'lookback': lookback,
                    'sharpe': sharpe,
                    'returns': rets,
                    'trade_log': log_,
                    'metrics': metrics,
                    'formation_start': window['formation_start'],
                    'formation_end': window['formation_end'],
                    'trading_start': window['trading_start'],
                    'trading_end': window['trading_end'],
                    'trading_length': len(rets)
                }

        if best_res is None:
            continue

        # 6.7) Armazena resultado deste período
        results.append(best_res)
        metrics_cache[cache_key] = best_res
        processed += 1

        # Atualiza séries globais
        dates = pd.date_range(best_res['trading_start'], periods=len(all_period_returns), freq='D')
        if all_period_returns:
            ser = pd.Series(all_period_returns, index=dates)
            all_returns = pd.concat([all_returns, ser]) if not all_returns.empty else ser
        all_trade_log.extend(best_res['trade_log'])

        # Salva cache periodicamente
        if period_idx % 5 == 0 or period_idx == len(windows) - 1:
            with open(METRICS_CACHE_FILE, 'wb') as f:
                pickle.dump(metrics_cache, f)
            with open(PAIRS_CACHE_FILE, 'wb') as f:
                pickle.dump(pair_cache, f)
            logging.info(f"Cache salvo após período {period_idx}")

        logging.info(
            f"Período {period_idx} concluído em {time.time() - start_time:.2f}s — "
            f"Trades: {total_trades}, Sharpe: {best_sharpe:.4f}"
        )

    # 7) Consolida resultados
    best_results = {r['period']: r for r in results}

    # 8) Salvar returns por período
    for period_idx, res in best_results.items():
        pd.DataFrame(
            res['returns'][:res['trading_length']],
            columns=['returns']
        ).to_csv(
            os.path.join(RESULTS_DIR, f"returns_period_{period_idx}.csv"),
            index=False
        )

    # 9) Calcula métricas globais
    if not all_returns.empty:
        global_metrics = calculate_metrics(all_returns, all_trade_log, rf_data)
        global_metrics.update({
            'daily_returns': all_returns,
            'daily_returns_index': all_returns.index,
            'equity_curve': (1 + all_returns).cumprod(),
            'cumulative_returns': (1 + all_returns).cumprod() - 1,
            'drawdowns': (1 + all_returns).cumprod() / (1 + all_returns).cumprod().cummax() - 1,
        })
    else:
        logging.warning("Nenhum retorno válido para cálculo de métricas globais")
        global_metrics = {
            'daily_returns': pd.Series(dtype=float),
            'daily_returns_index': pd.DatetimeIndex([]),
            'equity_curve': pd.Series(dtype=float),
            'cumulative_returns': pd.Series(dtype=float),
            'drawdowns': pd.Series(dtype=float),
        }

    # INCLUIR TRADE_LOG NO GLOBAL_METRICS (CORREÇÃO)
    global_metrics['trade_log'] = all_trade_log

    # 10) Salvar métricas resumo por período
    summary = []
    for idx, res in best_results.items():
        m = res['metrics']
        summary.append({
            'period': idx,
            'lookback': res['lookback'],
            'sharpe_ratio': m['sharpe_ratio'],
            'cumulative_return': m['cumulative_return'],
            'num_trades': m['num_trades'],
            'win_rate': m['win_rate'],
            'avg_trade_return': m['avg_trade_return']
        })

    # Garante pelo menos uma linha em df_summary
    if summary:
        df_summary = pd.DataFrame(summary)
    else:
        df_summary = pd.DataFrame([{
            'period': np.nan,
            'lookback': np.nan,
            'sharpe_ratio': np.nan,
            'cumulative_return': np.nan,
            'num_trades': np.nan,
            'win_rate': np.nan,
            'avg_trade_return': np.nan
        }])

    # Salva CSV usando somente df_summary
    df_summary.to_csv(
        os.path.join(RESULTS_DIR, "metrics_summary.csv"),
        index=False
    )

    logging.info(
        f"Processamento concluído: {processed}/{len(windows)} períodos em "
        f"{time.time() - start_time:.2f}s"
    )
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
        comp = global_metrics.get('crisis_comparison', {})
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
            logging.info(f"Diferença Sharpe       : {comp.get('sharpe_ratio_diff', 0):.2f}")
            logging.info(f"Outperformance absoluta: {comp.get('outperformance', 0)*100:.2f}%")

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
        for period_idx, res in best_results.items():
            msg = (
                f"Período {period_idx}: Lookback={res['lookback']}, "
                f"Sharpe={res['sharpe']:.4f}, "
                f"Retorno={res['metrics']['cumulative_return']*100:.2f}%, "
                f"Trades={res['metrics']['num_trades']}, "
                f"Win Rate={res['metrics']['win_rate']:.4f}"
            )
            logging.info(msg)

        logging.info(f"\nResultados salvos em {RESULTS_DIR}")
        logging.info(f"Tempo total de execução: {time.time()-start_time:.2f}s")

        # 11) Geração de gráficos e arquivos de saída
        try:
            # Cria DataFrame com métricas por período
            period_data = []
            for period_idx, res in best_results.items():
                period_data.append({
                    'period': period_idx,
                    'lookback': res['lookback'],
                    'sharpe_ratio': res['sharpe'],
                    'cumulative_return': res['metrics']['cumulative_return'],
                    'num_trades': res['metrics']['num_trades'],
                    'win_rate': res['metrics']['win_rate'],
                    'start_date': res['trading_start'],
                    'end_date': res['trading_end']
                })
            
            period_metrics = pd.DataFrame(period_data).sort_values('period')

            # Salva métricas por período em CSV
            period_metrics.to_csv(os.path.join(ANALYSIS_DIR, "period_metrics.csv"), index=False)
            
            # Gera gráficos
            plot_results(period_metrics, global_metrics, ANALYSIS_DIR)
            
            # Salva trades em CSV
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
    # main(use_cache=True)
