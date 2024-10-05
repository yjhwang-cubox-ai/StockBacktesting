# 필요한 라이브러리 임포트
import pandas as pd
from pykrx import stock
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import datetime

# 1. 삼성전자 주식 데이터를 pykrx 모듈로 불러오기 (2023년 1월부터 2024년 1월 15일까지)
start_date = "20230101"
end_date = "20241006"
ticker = "069080"  # 삼성전자 티커

start_date = "20150101"
end_date = "20241005"
ticker = '069080' # 웹젠 티커

# pykrx로 주식 데이터 불러오기
df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
df = df[['시가', '고가', '저가', '종가', '거래량']]  # 필요한 열만 선택

# 2. 주식 데이터 전처리
df['일자'] = df.index
df['다음날_고가'] = df['고가'].shift(-1)  # 다음날의 고가를 예측할 목표 변수로 설정
df.dropna(inplace=True)  # NaN 값 제거

# 특징(X)과 목표(y) 설정
X = df[['시가', '저가', '종가', '거래량']]
y = df['다음날_고가']

# 학습 데이터와 테스트 데이터 나누기 (2024년 1월 15일까지 학습, 이후 테스트)
# train_X = X[:'2024-01-15']
# train_y = y[:'2024-01-15']
# test_X = X['2024-01-16':]
# test_y = y['2024-01-16':]
train_X = X[:'2017-12-31']
train_y = y[:'2017-12-31']
test_X = X['2018-01-01':]
test_y = y['2018-01-01':]
# 3. 랜덤 포레스트 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)

# 4. 테스트 데이터로 예측
predictions = model.predict(test_X)

# 5. 예측 결과 시각화
plt.figure(figsize=(16,7))

# 실제 2023년 1월부터 2024년 1월 31일까지의 고가 데이터 시각화
plt.plot(df['일자'], df['다음날_고가'], label='Actual High Price', color='blue')

# 2024년 1월 16일부터 2024년 1월 31일까지의 예측값을 그래프에 추가
test_dates = df['일자']['2018-01-01':]
# test_dates = df['일자']['2024-01-16':]
plt.plot(test_dates, predictions, label='Predicted High Price (2024-01-16 ~ 2024-01-31)', color='red', linestyle='--')

# 그래프 설정
plt.title('Samsung Electronics Stock Price (High) Prediction')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# 그래프 출력
plt.show()
