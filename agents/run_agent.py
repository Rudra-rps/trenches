#!/usr/bin/env python3
"""
Trenches Agent Runner - Simplified main entry point using modular architecture.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to Python path to fix relative imports
sys.path.append(str(Path(__file__).parent))

# Import the new modular components
from core.simulation import TrenchesSimulation
from tools.onchain_tools import get_eth_balance

# Load environment variables
load_dotenv()


async def main():
    """Main entry point for the Trenches agent simulation"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Validate all required API keys
    if not os.getenv('GROQ_API_KEY'):
        logger.error("Groq API key not found in environment.")
        return
    logger.info("Groq API key found.")

    if not os.getenv('ETHERSCAN_API_KEY'):
        logger.error("Etherscan API key not found in environment.")
        return
    logger.info("Etherscan API key found.")

    config_dir = Path(os.getenv('CONFIG_DIR', 'config'))

    # Define the dictionary of available tools
    tools = {
        "get_eth_balance": get_eth_balance
    }
    
    try:
        # Initialize, register tools, and run simulation
        async with TrenchesSimulation(config_dir) as simulation:
            simulation.register_tools(tools)
            await simulation.run_simulation()

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())