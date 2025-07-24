import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def get_eth_balance(address: str) -> str:
    """Fetches the ETH balance for a given wallet address from Etherscan."""
    if not ETHERSCAN_API_KEY:
        return "Error: Etherscan API key not configured."

    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={ETHERSCAN_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data['status'] == '1':
            balance_wei = int(data['result'])
            balance_eth = balance_wei / 1e18
            return f"The balance of address {address} is {balance_eth:.4f} ETH."
        else:
            return f"Error from Etherscan API: {data.get('message', 'Unknown error')}"
    except requests.exceptions.RequestException as e:
        return f"An HTTP error occurred: {e}"