import os
import sys

# 1. Override the API URL environment variable to point to our local mock server
os.environ["API_URL"] = "http://127.0.0.1:8001"

# 2. Add the parent directory to the Python path so we can import agent.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import agent

if __name__ == "__main__":
    print(f"🔧 Environment Override: Pointing agent to MUST use LOCAL Mock Server at {os.environ['API_URL']}")
    # Start the agent as normal
    agent.run()
