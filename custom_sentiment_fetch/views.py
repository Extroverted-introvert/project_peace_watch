from django.shortcuts import render
from custom_sentiment_fetch.apps import CustomSentimentFetchConfig
from custom_sentiment_fetch.twitter_api2 import get_tweet_sentiment
import numpy as np

# Create your views here.
def custom_sentiment(request):
    context = {}
    print(request.method)
    
    if request.method == 'GET':
        return render(request, 'custom_sentiment/custom_sentiment.html', context)
    
    elif request.method == 'POST':
        # Check if user exists
        city = request.POST['city']
        query = request.POST['query']
        prediction_list =[]
        tweets = get_tweet_sentiment(city, query)
        for tweet in tweets:
            prepared_text=np.array([tweet.full_text])
            #prediction_score = np.float64(CustomSentimentFetchConfig.sentiment_model.predict(prepared_text)[0][0]).item()
            prediction_score = 0.5
            prediction_list.append(prediction_score)
        final_score = sum(prediction_list)/len(prediction_list)    
        prediction = assign_prediction(final_score)
        context ={'sentiment':prediction, 'tweet_list': tweets}
        return render(request, 'custom_sentiment/custom_sentiment.html', context)

def assign_prediction(prediction_score):
        pred_score=prediction_score*100
        sentiment = 'Negative'
        if(pred_score>=20 and pred_score<40):
            sentimentg = 'Poor'
        elif(pred_score>=40 and pred_score<60):
            sentiment = 'Average'
        elif(pred_score>=60 and pred_score<80):
            sentiment = 'Good'
        elif(pred_score>=80):
            sentiment = 'Positive' 
        return sentiment