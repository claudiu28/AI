import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import dotenv
dotenv.load_dotenv()

key = os.environ["AZURE_SUBSCRIPTION_KEY"]
endpoint = os.environ["AZURE_ENDPOINT"]

client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

message = "By choosing a bike over a car, I’m reducing my environmental footprint. Cycling promotes eco-friendly transportation, and I’m proud to be part of that movement."

result = client.analyze_sentiment(documents = [message], show_opinion_mining=True)
print(f"Overall sentiment: {result[0].sentiment}")