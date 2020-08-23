
import os
import sys
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))
os.chdir(os.path.realpath(os.path.dirname(__file__)))

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sqlite3

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from unidecode import unidecode
import time
from threading import Lock, Timer
import pandas as pd
from config import stop_words
import regex as re
from collections import Counter
import string
import pickle
import itertools
from textblob import TextBlob
import nltk
nltk.download('all')

analyzer = SentimentIntensityAnalyzer()

#consumer key, consumer secret, access token, access secret block. removed for privacy reasons.
ckey=csecret=atoken=asecret=""
# isolation lever disables automatic transactions,
#will insert from a sep thread
conn = sqlite3.connect('twitter.db', isolation_level=None, check_same_thread=False)
c = conn.cursor()

# Note to self, FTS is a mess.
def create_table():
    try:

        # concurrent r/wr
        c.execute("PRAGMA journal_mode=wal")
        c.execute("PRAGMA wal_checkpoint=TRUNCATE")
        #c.execute("PRAGMA journal_mode=PERSIST")

        c.execute("CREATE TABLE IF NOT EXISTS sentiment(id INTEGER PRIMARY KEY AUTOINCREMENT, unix INTEGER, tweet TEXT, sentiment REAL)")
        # key-value table for random stuff really.
        c.execute("CREATE TABLE IF NOT EXISTS misc(key TEXT PRIMARY KEY, value TEXT)")
        # DESC order
        c.execute("CREATE INDEX id_unix ON sentiment (id DESC, unix DESC)")
        # FTS 4.4.2;
        c.execute("CREATE VIRTUAL TABLE sentiment_fts USING fts5(tweet, content=sentiment, content_rowid=id, prefix=1, prefix=2, prefix=3)")
        # that trigger on insert, more on update and del
        c.execute("""
            CREATE TRIGGER sentiment_insert AFTER INSERT ON sentiment BEGIN
                INSERT INTO sentiment_fts(rowid, tweet) VALUES (new.id, new.tweet);
            END
        """)
    except Exception as e:
        print(str(e))
create_table()

# create a lock
lock = Lock()

class listener(StreamListener):

    data = []
    lock = None

    def __init__(self, lock):

        self.lock = lock

        #timer and super
        self.save_in_database()
        super().__init__()

    def save_in_database(self):

        # 1sec
        Timer(1, self.save_in_database).start()

        # transact if data.
        with self.lock:
            if len(self.data):
                c.execute('BEGIN TRANSACTION')
                try:
                    c.executemany("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)", self.data)
                except:
                    pass
                c.execute('COMMIT')

                self.data = []

    def on_data(self, data):
        try:
            #print('DATA T data')
            data = json.loads(data)
            # EX
            # {'limit': {'track': 14545, 'timestamp_ms': '1520219874522'}}
            if 'truncated' not in data:
                #print(data)
                return True
            if data['truncated']:
                tweet = unidecode(data['extended_tweet']['full_text'])
            else:
                tweet = unidecode(data['text'])
            time_ms = data['timestamp_ms']
            vs = analyzer.polarity_scores(tweet)
            sentiment = vs['compound']
            #print(time_ms, tweet, sentiment)

            # every 1 sec)
            with self.lock:
                self.data.append((time_ms, tweet, sentiment))

        except KeyError as e:
            #print(data)
            print(str(e))
        return True

    def on_error(self, status):
        print(status)


# make a counter with bl, empty, etc. and a regex.. same as before.
stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000] * len(stop_words))))

punctuation = [str(i) for i in string.punctuation]
split_regex = re.compile("[ \n" + re.escape("".join(punctuation)) + ']')

def map_nouns(col):
    return [word[0] for word in TextBlob(col).tags if word[1] == u'NNP']

# to generate "trending" data
def generate_trending():

    try:
        # select last max(10k) tweets
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10000", conn)
        df['nouns'] = list(map(map_nouns,df['tweet']))

        # convert to tokens
        tokens = split_regex.split(' '.join(list(itertools.chain.from_iterable(df['nouns'].values.tolist()))).lower())
        # clean and get top 10 (related)
        trending = (Counter(tokens) - blacklist_counter).most_common(10)

        # get sentiment vals
        trending_with_sentiment = {}
        for term, count in trending:
            df = pd.read_sql("SELECT sentiment.* FROM  sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 1000", conn, params=(term,))
            trending_with_sentiment[term] = [df['sentiment'].mean(), count]

        # save in a database
        with lock:
            c.execute('BEGIN TRANSACTION')
            try:
                c.execute("REPLACE INTO misc (key, value) VALUES ('trending', ?)", (pickle.dumps(trending_with_sentiment),))
            except:
                pass
            c.execute('COMMIT')


    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')
    finally:
        Timer(5, generate_trending).start()

Timer(1, generate_trending).start()

while True:

    try:
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener(lock))
        twitterStream.filter(track=["a","e","i","o","u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)
