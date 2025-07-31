import asyncio
from agents.tools.market_data_tools import get_liquidity_pool_info


async def test():
    result = await get_liquidity_pool_info("uniswap-v3", "ethereum")
    print(result)

if __name__ == "__main__":
    asyncio.run(test())
