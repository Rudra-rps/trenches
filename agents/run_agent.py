import time
import random
import yaml
from pathlib import Path

def load_agent_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def simulate_action(agent_id):
    actions = ["tweeted", "liked", "retweeted"]
    print(f"[{agent_id}] {random.choice(actions)} something...")

if __name__ == "__main__":
    config_files = Path("agent_spec").glob("*.yaml")
    for file in config_files:
        agent = load_agent_config(file)
        simulate_action(agent['id'])
        time.sleep(random.randint(1, 2))
