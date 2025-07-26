# tools/onchain_tools.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def get_eth_balance(address: str) -> dict:
    """
    Fetches the ETH balance for a given address and returns it as a dictionary.
    """
    if not ETHERSCAN_API_KEY:
        return {"error": "Etherscan API key not configured."}

    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={ETHERSCAN_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == '1':
            balance_wei = int(data['result'])
            # On success, return a dictionary with the data
            return {
                "address": address,
                "balance_eth": balance_wei / 1e18,
                "block_number": 0,  # Note: Free tier doesn't provide block number
                "error": None
            }
        else:
            # On API error, return a dictionary with an error message
            return {"error": data.get('message', 'Unknown Etherscan API error')}

    except requests.exceptions.RequestException as e:
        # On connection error, return a dictionary with an error message
        return {"error": f"An HTTP error occurred: {e}"}