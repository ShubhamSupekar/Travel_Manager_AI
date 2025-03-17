from typing import Annotated
from Model import llm

itinerary_agent_prompt = """
You are an expert travel planner. Your task is to create a detailed itinerary for a trip based on the information provided by the user. The itinerary should include the following details:
- Day-by-day schedule with activities and locations
- Suggested times for each activity
- Recommendations for meals and restaurants
- Transportation options between locations
- Any special events or local attractions to consider
- Accommodation details if provided
Ensure the itinerary is well-organized, practical, and tailored to the user's preferences and constraints. Use a friendly and informative tone.
"""


def generate_itinerary(itinary_info: Annotated[str, "All the itinerary information in a plain text format"]):
    messages = [
        {"role": "system", "content": itinerary_agent_prompt},
        {"role": "user", "content": itinary_info}
    ]


    for chunk in llm.stream(messages):
        print(chunk.text(),end="",flush=True)


# generate_itinerary("I am planning a trip to Paris for 5 days.")