import requests
import settings

def get_weather_by_city(city):
	params = {'q': city, 'appid': settings.OWM_API_KEY,
		'units': 'metric', 'lang': 'ru'}
	uri = 'http://api.openweathermap.org/data/2.5/weather'
	try:
		result = requests.get(uri, params=params)
		try:
			result.raise_for_status()
			return result.json()
		except (requests.HTTPError):
			return False
	except (requests.exceptions.RequestException):
		return False

if __name__ == "__main__":
    print(get_weather_by_city('Vladivostok,ru'))