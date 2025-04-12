# URL - https://app.prefect.cloud/account/8ff8f613-92c4-44ce-b811-f9956023e78d/workspace/04d8fca9-df2e-40c8-ae4f-a3733114c475/dashboard

# URL - https://app.prefect.cloud/api/docs

import requests

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "pnu_TxzjePDMkiA0EfGBQu1CnpQwJwhppK2hNRsn"  # Your Prefect Cloud API key
ACCOUNT_ID = "d416c2a3-6a1c-4723-b9ef-8a8a2afba338"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "ffb62f8a-0c79-4f98-a3fd-15d20c0197c1"  # Your Prefect Cloud Workspace ID
DEPLOYMENT_ID = "48aac213-0488-43f5-9f1c-1d44f5298f85"  # workflow.py deployment

# Correct API URL to get deployment details
PREFECT_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}/deployments/{DEPLOYMENT_ID}"

# Set up headers with Authorization
headers = {"Authorization": f"Bearer {PREFECT_API_KEY}"}

# Make the request using GET
response = requests.get(PREFECT_API_URL, headers=headers)

# Check the response status
if response.status_code == 200:
    deployment_info = response.json()
    print(deployment_info)
else:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")
