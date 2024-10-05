from pykrx import stock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 삼성전자 주식 데이터 불러오기 (2023년 1월 ~ 2023년 12월)
def get_stock_data(ticker, start_date, end_date):
    """
    주어진 티커의 주식 데이터를 pykrx를 사용해 가져옴.
    """
    df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
    df = df[['시가', '고가', '저가', '종가']]  # 필요한 컬럼만 선택
    df.columns = ['Open', 'High', 'Low', 'Close']  # 컬럼명을 영문으로 변경
    return df

# 삼성전자 티커: '005930'
# 셀트리온 티커: '068270'
ticker = '000440'
start_date = '20231101'
end_date = '20241004'

# 삼성전자 데이터를 가져옴
samsung_data = get_stock_data(ticker, start_date, end_date)

# 2. 변동성 돌파 전략 구현 함수
def breakout_strategy(data, k=0.5):
    """
    변동성 돌파 전략을 적용한 백테스팅 함수
    
    Parameters:
    - data: DataFrame with 'Open', 'High', 'Low', 'Close' columns.
    - k: 변동성 계수 (default = 0.5)
    
    Returns:
    - DataFrame: 거래 내역과 수익률 계산된 결과
    """
    # 전일 변동폭 계산
    data['range'] = data['High'].shift(1) - data['Low'].shift(1)
    
    # 매수 가격 계산 (전일 종가 + 변동폭 * k)
    data['buy_price'] = data['Open'] + data['range'] * k
    
    # 매수 조건: 당일 고가가 매수 가격을 넘으면 매수
    data['trade'] = np.where(data['High'] > data['buy_price'], 1, 0)
    
    # 수익률 계산 (매수가 발생한 날의 종가 기준 수익률)
    data['return'] = np.where(data['trade'] == 1, data['Close'] / data['buy_price'] - 1, 0)
    
    # 누적 수익률 계산
    data['cumulative_return'] = (1 + data['return']).cumprod()
    
    return data

# 3. 변동성 돌파 전략 적용 및 백테스팅
result = breakout_strategy(samsung_data, k=0.5)

# 4. 백테스팅 결과 출력
print(result.tail())  # 마지막 몇 줄을 출력하여 결과 확인
print(result)

# 1. 누적 수익률 차트
plt.figure(figsize=(10, 6))
plt.plot(result.index, result['cumulative_return'], label='Cumulative Return', color='blue', linewidth=2)
plt.title('Cumulative Return of Volatility Breakout Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.show()

# 2. 일일 수익률 차트
plt.figure(figsize=(10, 6))
plt.bar(result.index, result['return'], label='Daily Return', color='green')
plt.title('Daily Return of Volatility Breakout Strategy')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.grid(True)
plt.legend()
plt.show()

# 3. 매수 가격과 종가 비교 차트
plt.figure(figsize=(10, 6))
plt.plot(result.index, result['Close'], label='Close Price', color='red', linewidth=2)
plt.plot(result.index, result['buy_price'], label='Buy Price', color='blue', linestyle='--', linewidth=2)
plt.title('Close Price vs Buy Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()