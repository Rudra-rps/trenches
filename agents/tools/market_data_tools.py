# tools/market_data_tools.py
import requests
from typing import List

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