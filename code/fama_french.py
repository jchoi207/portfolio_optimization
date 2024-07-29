import yfinance as yf

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import sys
import os
import math

sys.path.append(os.path.abspath('code'))
import metrics


class fama_french():
    def __init__(self, path: str,  period: str = '1y', num_securities: int = 1, risk_free_rate: int = 0.0382, display_metrics: bool = False, predict_ret: bool =True):

        self.display_metrics = display_metrics
        self.predict_ret = predict_ret
        self.num_securities = num_securities - 1
        self.risk_free_rate = risk_free_rate
        self.period = period
        self.path = path
        
    def ff(self):
        self.initialize_attribs()
        self.get_beta_SMB_HML(display_metrics=self.display_metrics)
        return self.regression(predict_ret=self.predict_ret)

    def get_B_day(self, date: str = None, today: bool = False):

        def convert(date):
            if date.isoweekday() > 5:
                date -= BDay(1)
                return date.strftime("%Y-%m-%d")
            else:  # ignoring holidays for now
                return date.strftime("%Y-%m-%d")

        if today:
            date = pd.Timestamp.today().normalize()
            return convert(date)
        else:
            try:
                date = pd.Timestamp(date).normalize()
                return convert(date)
            except Exception as e:
                print(f"Error normalizing date {date} '{e}'")
                return

    def get_date_from_period(self, date):
        day_conv = {
            'd': 1,
            'mo': 30,
            'y': 365
        }
        substrs = list(day_conv.keys())
        op = [sub for sub in substrs if sub in self.period][0]
        n_days = int(int(self.period.split(op)[0])*day_conv[op])

        try:
            end_date = pd.to_datetime(date) - BDay(n_days)
            return self.get_B_day(date=end_date, today=False)

        except Exception as e:
            print("Error converting {date} to pd.DateTime '{e}'")
            return

    def get_tickers(self):
        self.path
        f = open(self.path, 'r')
        return [line.strip() for line in f]

    def initialize_attribs(self):
        self.tickers = self.get_tickers()
        self.today = self.get_B_day(today=True)
        self.today_year = int(self.today.split('-')[0])
        self.period_int = int(self.period.split('y')[0])
        self.start_ = self.today_year-self.period_int
        self.end_ = self.today_year + 1 # change this int to the window time frame
        self.corrupted_tickers = []

    def get_beta_SMB_HML(self, display_metrics: bool):
        # smb
        self.mc_rank = {}

        # hml
        self.bp_rank = {}

        # beta
        self.beta = {}

        columns = pd.MultiIndex.from_product(
            [self.tickers, ['R_i', 'Beta * (R_m - R_f)']], names=['Ticker', 'Metric'])
        self.all_returns = pd.DataFrame(
            columns=columns, index=range(self.start_, self.end_))
        print(f'>>Tickers: {self.tickers}')
        for year in range(self.start_, self.end_):
            start = f'{year}-01-01'
            end = f'{year}-12-31'

            if year == 2024:
                end = self.get_B_day(today=True)

            print(f'>>Current window: {start} {end}')
            
            # 5Y monthly beta
            mkt_ret = yf.Ticker('SPY').history(
                start=f'{year-5}-01-01', end=end, interval='1mo')['Close'].pct_change().dropna()

            beta_yearly = {}
            mc_yearly = {}
            se_yearly = {}
            for ticker in self.tickers:
                try:
                    stock = yf.Ticker(ticker)

                    num_shares = stock.info['sharesOutstanding']

                    history = stock.history(
                        start=start, end=end, interval='1mo')
                    history_beta_window = stock.history(
                        start=f'{year-5}-01-01', end=end, interval='1mo')

                    close_prices = history['Close']
                    ret = close_prices.pct_change().dropna()
                    ret_beta = history_beta_window['Close'].pct_change(
                    ).dropna()

                    # custom_bday = pd.offsets.CustomBusinessDay(holidays=holidays) TODO implement holidays (instead of just * 252)
                    num_B_days = len(pd.bdate_range(
                        pd.to_datetime(start), pd.to_datetime(end)))

                    exp_return = ret.mean() * (num_B_days / 21)

                    cov_matrix = np.cov(ret_beta, mkt_ret)

                    se_yearly[ticker] = metrics.get_bs_from_ticker(ticker=ticker,
                                                                   start=start,
                                                                   end=end,
                                                                   table=False,
                                                                   plot=False,
                                                                   stockholders_equity=True)[0]['stockholders_equity'].mean()

                    beta_yearly[ticker] = cov_matrix[0][1] / \
                        cov_matrix[1][1] * \
                        ((mkt_ret.mean()) - self.risk_free_rate)
                    mc_yearly[ticker] = [
                        math.log(np.array(close_prices * num_shares).mean()), exp_return]
                    se_yearly[ticker] = [
                        close_prices.mean() / (se_yearly[ticker] / num_shares), exp_return]
                    self.all_returns.loc[year, (ticker, 'R_i')] = exp_return
                    self.all_returns.loc[year,
                                         (ticker, 'Beta * (R_m - R_f)')] = beta_yearly[ticker]

                except Exception as e:
                    print(f">>Error {ticker}: '{e}'")
                    self.corrupted_tickers.append(ticker)
                    continue

            self.beta[year] = pd.DataFrame.from_dict(
                beta_yearly, orient='index', columns=['Beta'])
            self.mc_rank[year] = pd.DataFrame.from_dict(
                mc_yearly, orient='index', columns=['MC', 'Yearly Ret'])
            self.bp_rank[year] = pd.DataFrame.from_dict(
                se_yearly, orient='index', columns=['SE', 'Yearly Ret'])

        df_beta_columns = pd.MultiIndex.from_product(
            [range(self.start_, self.end_), ['Ticker', 'Beta']],  names=['Year', ''])
        self.df_beta = pd.DataFrame(columns=df_beta_columns)

        df_mc_columns = pd.MultiIndex.from_product(
            [range(self.start_, self.end_), ['Ticker', 'Yearly Ret']],  names=['Year', ''])
        self.df_mc = pd.DataFrame(columns=df_mc_columns)

        df_bp_columns = pd.MultiIndex.from_product(
            [range(self.start_, self.end_), ['Ticker', 'Yearly Ret']],  names=['Year', ''])
        self.df_bp = pd.DataFrame(columns=df_bp_columns)

        for year in range(self.start_, self.end_):
            self.beta[year] = self.beta[year]['Beta']
            self.df_beta[(year, 'Ticker')], self.df_beta[(
                year, 'Beta')] = self.beta[year].index, self.beta[year].values

            self.mc_rank[year] = self.mc_rank[year].sort_values(
                by=['MC'], ascending=False)['Yearly Ret']
            self.df_mc[(year, 'Ticker')], self.df_mc[(year, 'Yearly Ret')
                                                     ] = self.mc_rank[year].index, self.mc_rank[year].values

            self.bp_rank[year] = self.bp_rank[year].sort_values(
                by=['SE'], ascending=False)['Yearly Ret']
            self.df_bp[(year, 'Ticker')], self.df_bp[(year, 'Yearly Ret')
                                                     ] = self.bp_rank[year].index, self.bp_rank[year].values

        if display_metrics:
            print("\n>>Ticker betas w.r.t market")
            print(self.df_beta)
            print(">>Ticker returns sorted by ln(market_cap)")
            print(self.df_mc)
            print(">>Ticker returns sorted by P/B")
            print(self.df_bp)

        self.BETA = {}
        self.SMB = {}
        self.HML = {}
        self.reg = {}

        for year in range(self.start_, self.end_):
            midpoint = len(self.df_mc[(year, 'Yearly Ret')]) // 2
            self.SMB[year] = self.df_mc[(year, 'Yearly Ret')][midpoint:].mean(
                # small_cap - large_cap
            ) - self.df_mc[(year, 'Yearly Ret')][:midpoint].mean()
            self.HML[year] = self.df_bp[(year, 'Yearly Ret')][midpoint:].mean(
                # small_cap - large_cap
            ) - self.df_bp[(year, 'Yearly Ret')][:midpoint].mean()

        print(self.all_returns.transpose())

    def regression(self, predict_ret):
        def backtest_split(array, train_index):
            return np.array(array[:train_index]), np.array(array[train_index:])

        fama_french_predicted_returns = []
        train_index = self.period_int  # 1 testing size
    
        for ticker in self.tickers:
            try:
                r_train, r_test = backtest_split(
                    self.all_returns.transpose().loc[(ticker, 'R_i'), :], train_index)
                c_train, c_test = backtest_split(self.all_returns.transpose(
                ).loc[(ticker, 'Beta * (R_m - R_f)'), :], train_index)
                y = np.array(r_train - c_train - self.risk_free_rate)

                x_1_train, x_1_test = backtest_split(
                    np.array(list(self.SMB.values())), train_index)
                x_2_train, x_2_test = backtest_split(
                    np.array(list(self.HML.values())), train_index)
                X = np.vstack((x_1_train, x_2_train)).T

                model = LinearRegression()
                model.fit(X, y)

                beta_1, beta_2 = model.coef_
                epsilon = model.intercept_
                if self.display_metrics:
                    print('{}: beta_1: {:.2f}, beta_2: {:.2f}, epsilon: {:.2f}'.format(
                        ticker,
                        beta_1,
                        beta_2,
                        epsilon
                    ))
                    
                prediction = beta_1 * \
                    x_1_test[0] + beta_2 * x_2_test[0] + \
                    epsilon + c_test[0] + self.risk_free_rate
                ground_truth = r_test[0]
                    
                if predict_ret:
                    print(f'>> Predicting {self.end_-1} Yearly Returns for {ticker}')
                    fama_french_predicted_returns.append(prediction)
                    print('>> Prediction: {:.2f}'.format(prediction))
                else:        
                    print('Prediction: {:.2f} | Ground Truth: {:.2f}\n'.format(
                        prediction,
                        ground_truth
                    ))
                        
            except Exception as e:
                print(f">>Error {ticker}: '{e}'")
                continue
        return pd.DataFrame(data=fama_french_predicted_returns, index=self.tickers, columns=['FF Predicted Ret.'])

tickers_path = '/Users/jonathanchoi/Desktop/GitHub Projects/VIG-QS/portfolio-optimization/fama_french_tickers.md'

f = fama_french(path=tickers_path, 
                period='5y',
                num_securities=2,  
                display_metrics=False,
                predict_ret=True).ff()