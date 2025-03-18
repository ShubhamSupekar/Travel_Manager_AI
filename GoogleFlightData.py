import requests
import os

def fetch_flight_data(origin, destination, date, api_key):
    url = "https://serpapi.com/search"

    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "type": "2", # 1 = Round Trip, 2 = One Way
        "api_key": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "error" in data:
        print("Error:", data["error"])
        return
    
    if data['search_metadata']['status'] != 'Success':
        print("Failed to fetch flight data.")
    
    elif "best_flights" in data:
        all_flights_data = data["best_flights"]
        for flight_data in all_flights_data:
            if 'price' in flight_data:
                print("price: $",flight_data['price'])
            else:
                print("price: Not available")
                
            print("type: ",flight_data['type'])
            print("airplane: ",flight_data['flights'][0]['airplane'])
            print("airline: ",flight_data['flights'][0]['airline'])
            print("travel class: ",flight_data['flights'][0]['travel_class'])
            print("flight_number: ",flight_data['flights'][0]['flight_number'])
            print("duration: ",flight_data['total_duration'])
            print("departure airport name: ",flight_data['flights'][0]['departure_airport']['name'])
            print("departure airport id: ",flight_data['flights'][0]['departure_airport']['id'])
            print("departure airport time: ",flight_data['flights'][0]['departure_airport']['time'])
            print("arrival airport name: ",flight_data['flights'][0]['arrival_airport']['name'])
            print("arrival airport id: ",flight_data['flights'][0]['arrival_airport']['id'])
            print("arrival airport time: ",flight_data['flights'][0]['arrival_airport']['time'])
            print("____________________________________________________________________________________")
            print()
    else:
        print("No flights found.")


# Example usage
origin_city = "BOM"  # Example: Mumbai (IATA Code)
destination_city = "DEL"  # Example: Delhi (IATA Code)
travel_date = "2025-03-19"  # Format: YYYY-MM-DD
API_KEY = os.getenv('SERP_API_KEY')
fetch_flight_data(origin_city, destination_city, travel_date, API_KEY)
