# Import necessary libraries
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from datetime import datetime

# Load historical data for the S&P500 index
sp500_data = pd.read_csv("sp500.csv")
sp500_data = sp500_data.iloc[::-1]
# plt.plot(np.linspace(0,1, len(sp500_data["Returns"])), 0.01*sp500_data["Returns"],linewidth=0.3)
# plt.show()

# a standard GARCH(1,1) model
garch = arch_model(sp500_data["Returns"], vol='garch', p=1, o=0, q=1)
garch_fitted = garch.fit()

# # one-step out-of sample forecast
# garch_forecast = garch_fitted.forecast(horizon=1)
# predicted_et = garch_forecast.mean['h.1'].iloc[-1]
# print(predicted_et)


rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = sp500_data["Returns"][:-(test_size-i)]
    model = arch_model(train, p=1, q=1)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    
rolling_predictions = pd.Series(rolling_predictions, index=sp500_data["Returns"].index[-365:])

# fig,ax = plt.subplots(figsize=(10,4))
# ax.spines[['top','right']].set_visible(False)
# plt.plot(rolling_predictions)
# plt.title('Rolling Prediction')
# plt.show()


fig,ax = plt.subplots(figsize=(13,4))
ax.grid(which="major", axis='y', color='#758D99', alpha=0.3, zorder=1)
ax.spines[['top','right']].set_visible(False)
plt.plot(sp500_data["Returns"][-365:])
plt.plot(rolling_predictions)
plt.title('Tesla Volatility Prediction - Rolling Forecast')
plt.legend(['True Daily Returns', 'Predicted Volatility'])
plt.show()