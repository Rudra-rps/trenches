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

# Load environment variables
load_dotenv()


async def main():
    """Main entry point for the Trenches agent simulation"""

    # Setup logging
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Validate API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("Groq API key not found in environment")
        logger.info("Set your Groq API key: export GROQ_API_KEY=your_key_here")
        return
    else:
        # Only show first few chars for security
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        logger.info(f"Groq API key found: {masked_key}")

    # Set configuration directory
    config_dir = Path(os.getenv('CONFIG_DIR', 'config'))

    try:
        # Initialize and run simulation
        async with TrenchesSimulation(config_dir) as simulation:
            await simulation.run_simulation()

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        logger.info("Make sure the backend is running: cd backend && go run main.go")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f" Simulation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())