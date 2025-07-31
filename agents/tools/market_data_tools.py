# tools/market_data_tools.py
import requests
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

COINRANKING_API_KEY = os.getenv("COINRANKING_API_KEY")
API_HOST = "https://api.coinranking.com/v2"

def get_crypto_prices(coin_ids: List[str]) -> dict:
    """Fetches the current price of cryptocurrencies from CoinGecko."""
    ids = ",".join(coin_ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return {"prices": response.json(), "error": None}
    except requests.exceptions.RequestException as e:
        return {"error": f"An HTTP error occurred: {e}"}
    
def get_token_price(symbol: str) -> str:
    """
    Fetches the price of a given token symbol (e.g., 'ETH', 'BTC') from Coinranking.
    """
    if not COINRANKING_API_KEY:
        return "Error: Coinranking API key not configured."

    url = f"{API_HOST}/coins"
    headers = {'x-access-token': COINRANKING_API_KEY}
    params = {'symbols': [symbol]}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'success' and data['data']['coins']:
            coin = data['data']['coins'][0]
            price = float(coin['price'])
            return f"The current price of {coin['symbol']} is ${price:,.2f} USD."
        else:
            return f"Could not find price data for symbol '{symbol}'."

    except requests.exceptions.RequestException as e:
        return f"An HTTP error occurred: {e}"