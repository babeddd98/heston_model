import pandas as pd
import numpy as np
from yahoo_fin import options
import matplotlib.pyplot as plt
from datetime import datetime


def get_expiration_date(code: str, ticker: str) -> datetime:
    def manage_digits(dig):
        if dig[0] == 0: return dig[1]
        else: return dig
    
    len_ticker = len(ticker)
    if ticker[0] == "^":
        len_ticker -= 1
    year = int(manage_digits(code[len_ticker:len_ticker+2]))
    month = int(manage_digits(code[len_ticker+2:len_ticker+4]))
    day = int(manage_digits(code[len_ticker+4:len_ticker+6]))
    return datetime(2000 + year,month,day)


def get_market_data(ticker, option_type="call"):
    dates = options.get_expiration_dates(ticker)
    opt = []
    if option_type == "call":
        retrieve_function = options.get_calls
    elif option_type == "put":
        retrieve_function = options.get_puts
    else:
        retrieve_function = options.get_options_chain
    
    for date in dates:
        data = retrieve_function(ticker, date)
        data["Implied Volatility"] = [float(str(i).replace("%",""))for i in data["Implied Volatility"]]
        def get_current_date(current):
            current_date = current[:11].split("-")
            return datetime(int(current_date[0]), int(current_date[1]), int(current_date[2]))
        data["Date"] = [get_current_date(i) for i in data["Last Trade Date"]]
        data["Expiration"] = [get_expiration_date(i, ticker) for i in data["Contract Name"]]
        data["Time to Maturity"] = data["Expiration"] - data["Date"]
        opt.append(data)
    
    return opt


# calls = pd.DataFrame(assets["calls"])
# puts = pd.DataFrame(assets["puts"])
# # print(calls)
# calls["Implied Volatility"] = [float(str(i).replace("%",""))for i in calls["Implied Volatility"]]
# calls["Expiration"] = [get_expiration_date(i, len(ticker)) for i in calls["Contract Name"]]

# plt.show()
