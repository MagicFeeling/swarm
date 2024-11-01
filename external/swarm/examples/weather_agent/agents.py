# Import necessary classes for OpenAI Swarm
from swarm import Swarm, Agent
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# def get_weather(location, time="now"):
#     """Get the current weather in a given location. Location MUST be a city."""
#     return json.dumps({"location": location, "temperature": "65", "time": time})

# Web search function
def web_search(query):
    print(f"Performing web search for: {query}")
    print(f"TEST")
    return tavily_client.search(query)


def send_email(recipient, subject, body):
    print("Sending email...")
    print(f"To: {recipient}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    return "Sent!"


weather_agent = Agent(
    name="Weather Agent",
    model="llama3.2",
    instructions="You are a helpful agent who gives information of all kinds using web search.",
    functions=[web_search, send_email],
)
