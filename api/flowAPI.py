# URL - https://app.prefect.cloud/account/8ff8f613-92c4-44ce-b811-f9956023e78d/workspace/04d8fca9-df2e-40c8-ae4f-a3733114c475/dashboard

# URL - https://app.prefect.cloud/api/docs

import requests

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "pnu_R0dueb4K5KA2u0g87oSELGhFUKCC5L2Gs2P5"  # Your Prefect Cloud API key
ACCOUNT_ID = "d684a211-4820-4435-aed4-19f3074ca69c"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "0d6e4bdf-51f1-40df-961b-ec4ce037971a"  # Your Prefect Cloud Workspace ID
FLOW_ID = "61ec43b9-ba3e-418b-a0cb-81d9bc85e2ed"  # Your Flow ID for workflow.py

# Correct API URL to get flow details
PREFECT_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}/flows/{FLOW_ID}"

# Set up headers with Authorization
headers = {"Authorization": f"Bearer {PREFECT_API_KEY}"}

# Make the request using GET
response = requests.get(PREFECT_API_URL, headers=headers)

# Check the response status
if response.status_code == 200:
    flow_info = response.json()
    print(flow_info)
else:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")
