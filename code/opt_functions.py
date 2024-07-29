import pandas as pd
import numpy as np
import cvxpy as cp
import metrics as met
import matplotlib.pyplot as plt
import yfinance as yf


def optimize(universe: list, start_date: str, end_date: str, historical_returns: bool = True, min_max_weights: dict = None, short_selling: bool = False, risk_aversion: float = 0, **kwargs) -> list:
    """
    Optimize a portfolio using quadratic programming. Note this method is minimum variance. 
    The optimizer assumes any real number of stocks can be purchased (no ints)

    TODO:
        Implement sharpe ratio optimization
        Implement efficient return optimiation (requires casting)
        Come up with a better way to measure expected returns

    Args:
        universe (list): List of possible stock tickers to include in the portfolio. Note, optimizer may not include all tickers.
        start_date (str): The start date for the historical data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the historical data in 'YYYY-MM-DD' format.
        historical_returns (bool, default = True): Whether or not to base expected returns as mean of historical returns.
        min_max_weights (dict, optional): Dictionary where keys are tickers and values are lists 
                                          specifying the minimum and maximum weights for each ticker.
        short_selling (bool, optional): Whether to allow short-selling. If False, no short-selling is allowed.
        min_return (**kwargs, optional): Minimum expected return constraint as a decimal (e.g., 0.1 for 10%).
                                       Defaults to 0.1 (10%).
        risk_aversion (float, default = 0 ): Risk aversion factor for Markowitz mean variance optimization.

    Returns:
        opt_dict (dict): Optimal weights for each stock in the portfolio according to input constraints and objectives
        start_date (str): ####idk maybe remove dis
        end_date (str): ####idk maybe remove dis

    """
    n = len(universe)
    
    returns = []
    returns_matrix = pd.DataFrame()
    
    for ticker in universe:
        i = met.stock(ticker, start=start_date, end=end_date)
        returns.append(i.exp_ret())
        df = i.df
        returns_matrix[ticker] = df['Percent Change %']

    sigma = np.matrix(returns_matrix.cov())
    w = cp.Variable(n)

    A, b = np.ones((1, n)), np.array([1])
    G, h = -np.eye(n), np.zeros(n)
    expected_returns = np.array(returns)

    if short_selling:
        G *= -1
        
    constraints = [G @ w <= h, A @ w == b]
    
    if 'min_return' in kwargs:
        min_return = kwargs['min_return']
        min_return = np.array([min_return]) 
        constraints = [G @ w <= h, A @ w == b, expected_returns @ w >= min_return]
    
    if min_max_weights:
        for stock, weight in min_max_weights.items():
            idx = universe.index(stock)
            constraints.append(w[idx] >= weight[0])
            constraints.append(w[idx] <= weight[1])
    
    if not historical_returns:
        constraints = [G @ w <= h, A @ w == b]
        risk = 0
    
    if risk_aversion == 0:
        risk = 50
    else:
        risk = 1 / risk_aversion
    
    
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(w, sigma) - risk * expected_returns.T @ w), constraints)
    prob.solve()

    opt_weights = w.value
    expected_portfolio_return = expected_returns @ opt_weights
    expected_portfolio_variance = np.ravel(
        opt_weights.T @ sigma @ opt_weights).item()

    print(f"______________Summary Predictions______________\n")
    # for x, y in zip(universe, opt_weights):
    #     print(f"{x}: {round(y*100, 3)}%")

    pd.set_option('display.float_format', '{:.3f}'.format)
    print(pd.DataFrame(data=opt_weights*100, index=universe, columns=['Optimal Weights']))

    print(f"\nExpected monthly returns: {round(expected_portfolio_return, 2)}%")
    print(f"Portfolio st.dev: {round(expected_portfolio_variance, 2)} \n")

    opt_dict = {}
    for i in range(len(universe)):
        opt_dict[universe[i]] = list(opt_weights)[i]

    return opt_dict, start_date, end_date



def cmgr(self) -> float: 
    """
    Returns the compound annual (monthly) growth rate.
    
    Returns:
        float: CMGR
    """
    months = (pd.to_datetime(self.end) - pd.to_datetime(self.start)).days / 30.44 
    close_prices = np.array(self.df['Close']) 
    return (close_prices[-1]/close_prices[0]) ** (1/months) -1

    
def portfolio_performance(opt_dict: dict, start_date: str, end_date: str, principal: float = 1.00):
    """
    Calculate and plot the performance of a portfolio based on optimal weights.
    Also plot the performance of the individual instruments.
    Note this plot assumes that any real number of stocks can be purchased (no ints)

    Args:
        opt_dict (dict): Dictionary where keys are tickers and values are the optimal weights for each ticker.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        principal (float): Initial investment amount in USD.

    Returns: 
        None
    """
    price_data = {}

    for ticker in list(opt_dict.keys()):
        price_data[ticker] = yf.Ticker(ticker).history(
            start=start_date, end=end_date)['Close']

    df_prices = pd.DataFrame(price_data)
    df_prices = df_prices.div(df_prices.iloc[0])

    weights = pd.Series(opt_dict)
    df_weights = pd.DataFrame(weights, columns=['Weight']).T
    portfolio_value = df_prices.dot(df_weights.T).sum(axis=1) * principal

    plt.figure(figsize=(15, 8))
    plt.plot(portfolio_value.index, portfolio_value, label="Portfolio Value")


    last_close = []
    for ticker in df_prices.columns:

        plt.plot(df_prices.index, df_prices[ticker] *
                 principal, label=f'{ticker}', alpha=0.3)
    
    

    print(f"__________________Summary G.T__________________\n")
    time_interval = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    months = (time_interval).days / 30.44    
    
    
    cmgr = round(((portfolio_value.iloc[-1] / principal) ** ( 1 / months) -1)*100, 3)
    min_p, max_p = round(min(portfolio_value),2), round(max(portfolio_value),2)
    
    print(f"Realized monthly returns (CMGR): {cmgr}%")
    print(f"Simple rate of return: {round((portfolio_value.iloc[-1]-principal)/principal * 100, 2)}% over {round(time_interval.days/365, 2)} year(s)")
    print("\n  Low                    High")
    print(f"${min_p} <------------> ${max_p}")
    
    print(f"\nFinal realized value: ${round(portfolio_value.iloc[-1],2)}")
    
    
    plt.title(f"Optimal Portfolio Performance w/ Principal {principal}$")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
