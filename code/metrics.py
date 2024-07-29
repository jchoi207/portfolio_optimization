import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import requests


class stock:
    def __init__(self, ticker: str, start: str = "2024-01-01", end: str = "2024-05-30", i: str = "1mo") -> None:
        """ 
        Initialize the stock class with historical data.

        Args:
            ticker (str): Ticker symbol of the security.
            start (str): Start date for historical data in YYYY-MM-DD format. Default is "2024-01-01".
            end (str): End date for historical data in YYYY-MM-DD format. Default is "2024-05-30".
            i (str): Interval for historical data. Default is "1mo".
        """
        self._initialized = False
        self.ticker = ticker
        self.start = start
        self.end = end
        self.i = i

        self._load_data()

    def _load_data(self) -> None:
        """ 
        Load historical data and calculate the percentage change.
        """
        self.df = yf.Ticker(self.ticker).history(
            start=self.start, end=self.end, interval=self.i)
        self.df["Percent Change %"] = self.df["Close"].pct_change() * 100
        self.df.dropna(subset=["Percent Change %"], inplace=True)
        self.dividend_5 = yf.Ticker(self.ticker).info.get(
            "fiveYearAvgDividendYield")
        self.name = yf.Ticker(self.ticker).info.get("shortName")

    def st_dev(self) -> float:
        """
        Calculate and return the standard deviation of returns.

        Returns:
            float: Standard deviation of returns.
        """
        st_deviation = round(np.std(self.df["Percent Change %"]), 2)
        return st_deviation

    def exp_ret(self) -> float:
        """
        Calculate and return the expected value of historical returns.

        Returns:
            float: Expected value of historical returns.
        """
        exp_returns = round(np.average(self.df["Percent Change %"]), 2)
        return exp_returns

    def coeff_var(self) -> float:
        """
        Calculate and return the coefficient of variation.

        Returns:
            float: Coefficient of variation.
        """
        self._load_data()
        coeff_variation = round(self.st_dev() / self.exp_ret(), 2)
        return coeff_variation

    def beta(self, benchmark: str = "VOO") -> float:
        """
        Calculate and return the beta of self.ticker relative to the benchmark.

        Args:
            benchmark (str): Ticker symbol of the benchmark security. Default is "VOO".

        Returns:
            float: Beta of the security relative to the benchmark.
        """
        # create an instance for benchmark
        bmark = stock(benchmark, self.start, self.end)

        # merge dfs to not overwrite
        merged_df = pd.merge(self.df["Percent Change %"], bmark.df["Percent Change %"],
                             left_index=True, right_index=True, suffixes=("_stock", "_benchmark"))
        cov_matrix = np.cov(
            merged_df["Percent Change %_stock"], merged_df["Percent Change %_benchmark"])

        beta = round(cov_matrix[0][1] / cov_matrix[1][1], 2)

        return beta

    def all(self) -> list:
        """
        Return all calculated metrics in a list.

        Returns:
            list: A list containing the name, ticker, expected returns, standard deviation, 
                  coefficient of variation, and beta of the security.
        """
        return self.name, self.ticker, self.exp_ret(), self.st_dev(), self.coeff_var(), self.beta()


def create_df_INDUSTRY(filename: str, start: str, end: str) -> pd.DataFrame:
    """
    Create a DataFrame with metrics for each stock in the specified industries.

    Args:
        filename (str): Path to the file containing tickers by industry.
        start (str): Start date for historical data.
        end (str): End date for historical data.

    Returns:
        pd.DataFrame: DataFrame containing sector, name, ticker, expected returns, 
                      standard deviation, coefficient of variation, and beta for each stock.
    """
    with open(filename, "r") as file:
        tickers_dict = {
            "Tech": [],
            "Enrgy": [],
            "Fin": [],
            "Heal": []
        }
        for line in file:
            industry, ticker = line.strip().split(" ")
            tickers_dict[industry].append(ticker)
    industries = list(tickers_dict.keys())

    df = pd.DataFrame(columns=["Sector", "Name", "Ticker", "E[Returns]",
                               "StDev", "Coeff Var", "Beta"])
    k = 0
    for i in range(len(industries)):
        for j in range(len(tickers_dict[industries[i]])):
            df.loc[k] = [industries[i]] + [tickers_dict[industries[i]][j]] + \
                list(stock(tickers_dict[industries[i]][j], start, end).all())
            k += 1
    return df


def create_df(filename: str, start: str, end: str) -> pd.DataFrame:
    """
    Create a DataFrame with metrics for each stock in the provided file.

    Args:
        filename (str): Path to the file containing tickers.
        start (str): Start date for historical data.
        end (str): End date for historical data.

    Returns:
        pd.DataFrame: DataFrame containing name, ticker, expected returns, 
                      standard deviation, coefficient of variation, and beta for each stock.
    """
    tickers = []
    with open(filename, "r") as file:
        for line in file:
            tickers.append(line.strip())
    print(tickers)
    df = pd.DataFrame(columns=["Name", "Ticker", "E[Returns]",
                               "StDev", "Coeff Var", "Beta",])

    for i in range(len(tickers)):
        df.loc[i] = list(stock(tickers[i], start, end).all())

    return df


def sort(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Sort the DataFrame based on the specified column.

    Args:
        df (pd.DataFrame): DataFrame to be sorted.
        column (str): Column name to sort by.

    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    return df.sort_values(column, ascending=True)


# Example Call of YTD data ranking securities by standard deviation
# Replace MY_FILE.txt with your own file containing tickers separated by line breaks
# print(sort(create_df("MY_FILE.txt",
#                      "2024-01-01", "2024-06-14"), "StDev").dropna())


def get_bs_from_ticker(ticker: str, start: str, end: str, table: bool = True, plot: bool = True, stockholders_equity: bool = False):
    """
    Fetch and return balance sheet metrics for a given ticker from the SEC.

    Args:
        ticker (str): Ticker symbol of the company.
        start (str): Start date for the data in YYYY-MM-DD format.
        end (str): End date for the data in YYYY-MM-DD format.
        table (bool): Whether to display the table of metrics. Default is True.
        plot (bool): Whether to plot the metrics over time. Default is True.
        stockholders_equity (bool): Whether to fetch only stockholders' equity. Default is False.

    Returns:
        pd.DataFrame: DataFrame with the balance sheet metrics for the specified period.
    """
    headers = {'User-Agent': "currymachinchin@gmail.com"}

    all_tickers = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers
    )
    company_data = pd.DataFrame.from_dict(all_tickers.json(), orient='index')
    company_data['cik_str'] = company_data['cik_str'].astype(str).str.zfill(10)

    try:
        my_cik = company_data[company_data['ticker']
                              == ticker.upper()]['cik_str'].iloc[0]
    except:
        print(f"Ticker {ticker} not found in database.")
        return

    root = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{my_cik}/us-gaap'

    urls = {
        "stockholders_equity": f'{root}/StockholdersEquity.json',
    }

    if not stockholders_equity:
        urls.update({
            "total_assets": f'{root}/Assets.json',
            "total_liabilities": f'{root}/Liabilities.json',
            "current_assets": f'{root}/AssetsCurrent.json',
            "current_liabilities": f'{root}/LiabilitiesCurrent.json',
            # "revenues": f'{root}/Revenues.json',
            "net_income_loss": f'{root}/NetIncomeLoss.json'
        })

    data = {}
    for metric, url in urls.items():
        response = requests.get(url, headers=headers)
        data[metric] = pd.DataFrame.from_dict(response.json()['units']['USD'])

    def process_metric(metric_data, key):
        metric_data = metric_data[metric_data.form == '10-Q']
        metric_data = metric_data.reset_index(drop=True)
        metric_data = metric_data[['end', 'val']]
        metric_data['val'] = metric_data['val'] / 1e0
        metric_data.columns = ["Date", f"{str(key)}"]
        # metric_data.set_index('Date', inplace=True)
        return metric_data
    lengths = []
    for key, value in data.items():
        data[key] = process_metric(value, key)
        lengths.append(data[key].shape[0])
        # data[key] = data[key].reset_index()

    first_key = list(data.keys())[0]
    merged_df = data[first_key].copy()
    # print(merged_df)
    for key in list(data.keys())[1:]:
        df = data[key]
        merged_df = pd.merge(merged_df, df, on='Date',
                             how='outer', suffixes=('', f'_{key}'))
    merged_df = merged_df.set_index('Date')
    merged_df = merged_df.ffill().bfill()
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    pd.set_option('display.float_format', '{:.2e}'.format)
    # print(f"Balance Sheet Metrics for ticker {ticker}")
    merged_df.sort_index(axis=1, ascending=True)

    if table:
        print(merged_df.head(10))
    if plot:
        plt.figure(figsize=(10, 8))

        for column in merged_df.columns:
            plt.plot(merged_df.index, merged_df[column], label=column)

        # Customize the plot
        plt.title(f'Financial Metrics Over Time for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate labels if necessary
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.show()
    merged_df_date = merged_df[(merged_df.index >= start)
                               & (merged_df.index <= end)]
    return merged_df_date, merged_df
