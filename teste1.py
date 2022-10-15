import pandas as pd
import GetOldTweets3 as got

nro_tweets=500
tweetCriteria = got.manager.TweetCriteria().setQuerySearch('ITUB4')\
                                           .setSince("2019-12-01")\
                                           .setUntil("2020-04-25")\
                                           .setMaxTweets(nro_tweets)
                                         

tweet_text = got.manager.TweetManager.getTweets(tweetCriteria)
    
for t in range(0, nro_tweets -1):
    tweet_text[t] = tweet_text[t].text


df = pd.DataFrame(tweet_text)

csv_data = df.to_csv('2019-12_2020-04_ITUB4_500',index=False)

print("Funcionou")      