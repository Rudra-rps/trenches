# tools/orderbook.py

def get_order_book(token_pair: str) -> str:
    # Simulated order book snapshot
    sample_order_books = {
        "ETH/USDC": {
            "bids": [(2000, 5), (1995, 10), (1990, 8)],
            "asks": [(2005, 4), (2010, 7), (2020, 6)]
        }
    }

    book = sample_order_books.get(token_pair.upper())
    if not book:
        return f"No order book data found for {token_pair}."

    bid_str = "\n".join([f" - ${price} × {amount} ETH" for price, amount in book["bids"]])
    ask_str = "\n".join([f" - ${price} × {amount} ETH" for price, amount in book["asks"]])

    return f"Order Book for {token_pair}:\nBids:\n{bid_str}\nAsks:\n{ask_str}"
