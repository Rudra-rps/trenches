import httpx
import logging

async def get_liquidity_pool_info(protocol_slug: str = "uniswap-v3", chain: str = "ethereum") -> dict:
    """
    Fetch real-time liquidity pool data from DeFiLlama for a specific protocol.
    Args:
        protocol_slug (str): e.g. 'uniswap-v3', 'curve', 'sushiswap'
        chain (str): e.g. 'ethereum', 'polygon', 'arbitrum'
    Returns:
        dict: Top 5 pools with TVL, tokens, APY, etc.
    """
    try:
        url = f"https://yields.llama.fi/pools"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        
        pools = [
            pool for pool in data["data"]
            if pool.get("project") == protocol_slug and pool.get("chain", "").lower() == chain.lower()
        ]
        
        top_pools = sorted(pools, key=lambda x: x.get("tvlUsd", 0), reverse=True)[:5]

        result = []
        for p in top_pools:
            result.append({
                "pool": p.get("pool"),
                "symbol": p.get("symbol"),
                "tvl_usd": round(p.get("tvlUsd", 0), 2),
                "apy": round(p.get("apy", 0), 3),
                "project": p.get("project"),
                "chain": p.get("chain"),
                "url": p.get("url", "N/A"),
            })

        return {"protocol": protocol_slug, "chain": chain, "top_pools": result}

    except Exception as e:
        logging.exception("Failed to fetch liquidity pool info")
        return {"error": str(e)}
