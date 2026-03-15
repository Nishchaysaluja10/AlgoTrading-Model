import requests

class APIHandler:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def fetch_market_data(self):
        """Pulls the latest market tick from the server."""
        try:
            # TODO: Update this endpoint url tomorrow morning
            endpoint = f"{self.base_url}/market-data"
            response = requests.get(endpoint, headers=self.headers, timeout=5)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API Fetch Error (Network/Server): {e}")
            return None

    def execute_trade(self, action, amount):
        """Sends the BUY/SELL order to the server."""
        payload = {
            "action": action, 
            "amount": amount
        }
        try:
            # TODO: Update this endpoint url tomorrow morning
            endpoint = f"{self.base_url}/trade"
            response = requests.post(endpoint, json=payload, headers=self.headers, timeout=5)
            response.raise_for_status()
            
            print(f"✅ Trade Executed Successfully: {action} | Amount: {amount}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Trade Execution Failed: {e}")
            return None