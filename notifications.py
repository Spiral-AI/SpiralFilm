import os
from slack_sdk import WebClient

def notify_slack(message):
    client = WebClient(token=os.getenv("SLACK_API_TOKEN"))
    client.chat_postMessage(channel="#alerts", text=message)
