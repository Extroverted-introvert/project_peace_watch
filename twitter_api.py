import requests
import json
#its bad practice to place your bearer token directly into the script (this is just done for illustration purposes)
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAJcqTwEAAAAAkza3KKNMohwBGHcd4%2Filvq2vUYM%3D8IeDMZsIcGtRQYEIGxGT4QilMSXvDSAbOAwyJdq8IZ0kMqy2DH"
#define search twitter function
def search_twitter(query, tweet_fields, bearer_token = BEARER_TOKEN):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}

    url = "https://api.twitter.com/2/tweets/search/recent?query={}&{}".format(
        query, tweet_fields
    )
    response = requests.request("GET", url, headers=headers)

    print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

json_response = search_twitter(query="India", tweet_fields="tweet.fields=text")
print(json.dumps(json_response, indent=4, sort_keys=True))