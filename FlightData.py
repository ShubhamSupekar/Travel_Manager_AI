import requests
import os

# Get the flights details from the api 
def fetch_flight_data(api_key, dep_iata, arr_iata):
    base_url = 'http://api.aviationstack.com/v1/flights'
    params = {
        'access_key': api_key,
        'dep_iata': dep_iata,
        'arr_iata': arr_iata
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data:
            return data['data']
        else:
            print("No data found in the response.")
            return None
    else:
        print(f"Error: {response.status_code}")
        return None

# Add all flight details into list
def format_flights(flights):
    flight_info = []
    for flight in flights:
        try:
            airline = flight['airline']['name']
            flight_number = flight['flight']['number']
            departure_airport = flight['departure']['airport']
            departure_time = flight['departure']['scheduled']
            arrival_airport = flight['arrival']['airport']
            arrival_time = flight['arrival']['scheduled']
            flight_info.append(
                f"Flight {flight_number} by {airline} from {departure_airport} at {departure_time} to {arrival_airport} at {arrival_time}."
            )
            print(f"Flight {flight_number} by {airline} from {departure_airport} at {departure_time} to {arrival_airport} at {arrival_time}.")
        except KeyError:
            continue

    return flight_info if flight_info else "No valid flights found."


if __name__ == "__main__":
    # User Inputs
    source = input("Enter your departure city (IATA Code): ")
    destination = input("Enter your destination (IATA Code): ")
    
    # start_date = input("Enter your journey start date (YYYY-MM-DD): ")
    # end_date = input("Enter your journey end date (YYYY-MM-DD): ")
    
    # API Key
    API_KEY = os.getenv('aviationAPIkey')
    flights = fetch_flight_data(API_KEY,source,destination)

    # Get the flights information in list
    flights = format_flights(flights)