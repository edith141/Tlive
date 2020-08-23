# Tlive
Live sentiment analysis and insights for twitter. Built using Python.

WHAT FILES DO?

- `sents.py` - This is currently the main front-end application code. app layouts, logic for graphs & tables, interfaces with the db, etc.. This code is setup to run on a flask instance. Use the `dev_server.py` to run locally!

- `dev_server.py` - Runs this application, on the dev server.

- `twitter_stream.py` - Should be running in the background before strating main app. This is what get the tweet streams from Twitter, stores them into the db.

- `config.py` - Meant for many configurations in fututre, rn it just contains stop words. May change in future!

- `cache.py` -  For caching purposes to get and set things faster.

- `db-truncate.py` - Script truncates the literally infinitely-growing db. Gets ~3.5 millionish tweets per day if left to run, depending on how fast your system can process. No harm in keeping the db as is, but as the database grows, search times will suffer and no point storing old tweets (older than 2-3 days) since system is realtime. 

HOW TO START? Do the following in order!

- install everything liste in req.txt using pip (python). `pip install -r req.txt` should do it via terminal. Insure PIP and Python 3.x is installed.

IMP:- twitter_stream.py WILL DOWNLOAD REQ FILES FOR NLP. extracted size would be around ~3.3 GB!

- run twitter_stream.py to create DB and have it run all the time.

- finally run dev_server.py to start.

[ctrl+c to stop server and twitter_stream.py individually].
[Run twitter_stream.py in different cmd/terminal window and dev_server.py in diff window cmd/terminal window].

WINDOWS? READ BELOW!

- May need to latest version of sqlite db (ver 3?)
- Will need to configure FTS on sqlite3. Needs some binary files to be compiled. Did it before on my machine, don't remember exactly where I put the files. Stackoverflow helps. Will upload the binary files. 
