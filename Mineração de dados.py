import pandas as pd
import GetOldTweets3 as got

nro_tweets=200
tweetCriteria = got.manager.TweetCriteria().setUsername("jairbolsonaro")\
                                           .setSince("2020-09-10")\
                                           .setUntil("2020-09-15")\
                                           .setMaxTweets(nro_tweets)
                                         

tweet_text = got.manager.TweetManager.getTweets(tweetCriteria)
    
for t in range(0, nro_tweets -1):
   tweet_text[t] = tweet_text[t].text


df = pd.DataFrame(tweet_text)

csv_data = df.to_csv('2020_01 a 09_20_cafecomferri_1018',index=False)

print(tweet_text.text)     
