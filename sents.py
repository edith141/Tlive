# set chdir to curr dir //idk, may modify when hosting it on some configured server?
import os
import sys
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))
os.chdir(os.path.realpath(os.path.dirname(__file__)))
 
# other imports
import dash
from dash.dependencies import Output, Event, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import sqlite3
import pandas as pd
from collections import Counter
import string
import regex as re
from cache import cache
from config import stop_words
import time
import pickle

# will use only one sql connection throughout. 
# Only selects are used, no need for serialization or other things.
conn = sqlite3.connect('twitter.db', check_same_thread=False)
punctuation = [str(i) for i in string.punctuation]


# color schemes. //May change later.
sent_color_scheme = {-1:"#EE6055",
                    -0.5:"#FDE74C",
                     0:"#FFE6AC",
                     0.5:"#D0F2DF",
                     1:"#9CEC5B",}

app_color_scheme = {
    'background': '#0C0F0A',
    'text': '#FFFFFF',
    'sentiment-plot':'#41EAD4',
    'volume-bar':'#FBFC74',
    'someothercolor':'#FF206E',
}

POS_NEG_NEUT = 0.1

MAX_DF_LENGTH = 100

app = dash.Dash(__name__)

# main app layout. //not the best, but works.
app.layout = html.Div(
    [   html.Div(className='container-fluid edith', children=[html.H2('Live Sentiment (Twitter)', style={'color':"#CECECE"}),
                                                        html.H5('Search term:', style={'color':app_color_scheme['text']}),
                                                  dcc.Input(id='sentiment_term', value='twitter', type='text', style={'color':app_color_scheme['someothercolor']}),
                                                  ],
                 style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000}),

        
        
        html.Div(className='row edith', children=[html.Div(id='related-sentiment', children=html.Button('Loading related terms...', id='related_term_button'), className='col s12 m6 l6', style={"word-wrap":"break-word"}),
                                            html.Div(id='recent-trending', className='col s12 m6 l6', style={"word-wrap":"break-word"})]),

        html.Div(className='row edith', children=[html.Div(dcc.Graph(id='live-graph', animate=False), className='col s12 m6 l6'),
                                            html.Div(dcc.Graph(id='historical-graph', animate=False), className='col s12 m6 l6')]),

        html.Div(className='row edith', children=[html.Div(id="recent-tweets-table", className='col s12 m6 l6'),
                                            html.Div(dcc.Graph(id='sentiment-pie', animate=False), className='col s12 m6 l6'),]),
        
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),
        dcc.Interval(
            id='historical-update',
            interval=60*1000
        ),

        dcc.Interval(
            id='related-update',
            interval=30*1000
        ),

        dcc.Interval(
            id='recent-table-update',
            interval=2*1000
        ),

        dcc.Interval(
            id='sentiment-pie-update',
            interval=60*1000
        ),

    ], style={'backgroundColor': app_color_scheme['background'], 'margin-top':'-30px', 'height':'2000px',},
)


def df_resample_sizes(df, maxlen=MAX_DF_LENGTH):
    df_len = len(df)
    resample_amt = 100
    vol_df = df.copy()
    vol_df['volume'] = 1

    ms_span = (df.index[-1] - df.index[0]).seconds * 1000
    rs = int(ms_span / maxlen)

    df = df.resample('{}ms'.format(int(rs))).mean()
    df.dropna(inplace=True)

    vol_df = vol_df.resample('{}ms'.format(int(rs))).sum()
    vol_df.dropna(inplace=True)

    df = df.join(vol_df['volume'])

    return df

# make a counter with the blacklist words and empty words with some big value
# will use later to filter counter maybe.
stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000]*len(stop_words))))

# regex (punctuation list, space, nl)
split_regex = re.compile("[ \n"+re.escape("".join(punctuation))+']')

# find related words
def related_sentiments(df, sentiment_term, how_many=15):
    try:

        related_words = {}

        tokens = split_regex.split(' '.join(df['tweet'].values.tolist()).lower())

        # remove stop_words, sentiment_term and empty token by just making another counter
        # with some big value and substracting it (remove tokens w/ -ve count)
        blacklist_counter_with_term = blacklist_counter.copy()
        blacklist_counter_with_term[sentiment_term] = 1000000
        counts = (Counter(tokens) - blacklist_counter_with_term).most_common(15)

        for term,count in counts:
            try:
                df = pd.read_sql("SELECT sentiment.* FROM  sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 200", conn, params=(term,))
                related_words[term] = [df['sentiment'].mean(), count]
            except Exception as e:
                with open('errors.txt','a') as f:
                    f.write(str(e))
                    f.write('\n')

        return related_words

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


def fast_color(s):
    # don't delete. may use later.
    # except return bg as app_color_scheme['background']
    if s >= POS_NEG_NEUT:
        # positive
        return "#002C0D"
    elif s <= -POS_NEG_NEUT:
        # negative:
        return "#270000"

    else:
        return app_color_scheme['background']

def generate_html_table(df, max_rows=10):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color':app_color_scheme['text']}
                                  )
                              ),
                          html.Tbody(
                              [
                                  
                              html.Tr(
                                  children=[
                                      html.Td(data) for data in d
                                      ], style={'color':app_color_scheme['text'],
                                                'background-color':fast_color(d[2])}
                                  )
                               for d in df.values.tolist()])
                          ]
    )


# essentially a yes/no func to get right col based on -ve +ve
def pos_neg_neutral(col):
    if col >= POS_NEG_NEUT:
        # positive
        return 1
    elif col <= -POS_NEG_NEUT:
        # negative:
        return -1

    else:
        return 0
    
            
@app.callback(Output('recent-tweets-table', 'children'),
              [Input(component_id='sentiment_term', component_property='value')],
              events=[Event('recent-table-update', 'interval')])        
def update_recent_tweets(sentiment_term):
    if sentiment_term:
        df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10", conn, params=(sentiment_term+'*',))
    else:
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10", conn)

    df['date'] = pd.to_datetime(df['unix'], unit='ms')

    df = df.drop(['unix','id'], axis=1)
    df = df[['date','tweet','sentiment']]

    return generate_html_table(df, max_rows=10)

# configure the overall pie graph
@app.callback(Output('sentiment-pie', 'figure'),
              [Input(component_id='sentiment_term', component_property='value')],
              events=[Event('sentiment-pie-update', 'interval')])
def update_pie_chart(sentiment_term):

    # get data from cache if possible
    for i in range(100):
        sentiment_pie_dict = cache.get('sentiment_shares', sentiment_term)
        if sentiment_pie_dict:
            break
        time.sleep(0.1)

    if not sentiment_pie_dict:
        return None

    labels = ['Positive','Negative']

    try: pos = sentiment_pie_dict[1]
    except: pos = 0

    try: neg = sentiment_pie_dict[-1]
    except: neg = 0

    
    
    values = [pos,neg]
    colors = ['#007F25', '#800000']

    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value', 
                   textfont=dict(size=20, color=app_color_scheme['text']),
                   marker=dict(colors=colors, 
                               line=dict(color=app_color_scheme['background'], width=2)))

    return {"data":[trace],'layout' : go.Layout(
                                                  title='Positive vs Negative sentiment for "{}" (longer-term)'.format(sentiment_term),
                                                  font={'color':app_color_scheme['text']},
                                                  plot_bgcolor = app_color_scheme['background'],
                                                  paper_bgcolor = app_color_scheme['background'],
                                                  showlegend=True)}



# rt graph. updates in ~realtime
@app.callback(Output('live-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value')],
              events=[Event('graph-update', 'interval')])
def update_graph_scatter(sentiment_term):
    try:
        if sentiment_term:
            df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 1000", conn, params=(sentiment_term+'*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 1000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        init_length = len(df)
        df['smoothed_sentiment_vals'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df = df_resample_sizes(df)
        X = df.index
        Y = df.smoothed_sentiment_vals.values
        Y2 = df.volume.values
        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                line = dict(color = (app_color_scheme['sentiment-plot']),
                            width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_color_scheme['volume-bar']),
                )

        return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right'),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                                          title='Live sentiment for: "{}"'.format(sentiment_term),
                                                          font={'color':app_color_scheme['text']},
                                                          plot_bgcolor = app_color_scheme['background'],
                                                          paper_bgcolor = app_color_scheme['background'],
                                                          showlegend=False)}

    # try to get the error. helps sometimes. 
    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

# lt graph.
@app.callback(Output('historical-graph', 'figure'),
              [Input(component_id='sentiment_term', component_property='value'),
               ],
              events=[Event('historical-update', 'interval')])
def update_hist_graph_scatter(sentiment_term):
    try:
        if sentiment_term:
            df = pd.read_sql("SELECT sentiment.* FROM sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 10000", conn, params=(sentiment_term+'*',))
        else:
            df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10000", conn)
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        # saving this to a file brfoe going ahead due to plotly restrictions.
        # multiple-outputs-from-single-input-with-one-callback/4970. was in documentaion.
        # dash?plotly

        # store related sentiments in cache first.
        cache.set('related_terms', sentiment_term, related_sentiments(df, sentiment_term), 120)

        #print('rel-sent func called w/ terms {df} {sentiment_term}')
        #print(related_sentiments(df,sentiment_term), sentiment_term)
        init_length = len(df)
        df['smoothed_sentiment_vals'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df.dropna(inplace=True)
        df = df_resample_sizes(df,maxlen=500)
        X = df.index
        Y = df.smoothed_sentiment_vals.values
        Y2 = df.volume.values

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                line = dict(color = (app_color_scheme['sentiment-plot']),
                            width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color=app_color_scheme['volume-bar']),
                )

        df['sentiment_shares'] = list(map(pos_neg_neutral, df['sentiment']))

        # no need for dict now. may change later.
        #sentiment_shares = dict(df['sentiment_shares'].value_counts())
        cache.set('sentiment_shares', sentiment_term, dict(df['sentiment_shares'].value_counts()), 120)

        return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]), # add type='category to remove gaps'
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right'),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                                          title='Longer-term sentiment for: "{}"'.format(sentiment_term),
                                                          font={'color':app_color_scheme['text']},
                                                          plot_bgcolor = app_color_scheme['background'],
                                                          paper_bgcolor = app_color_scheme['background'],
                                                          showlegend=False)}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')



max_size_change = .4

def generate_size(value, smin, smax):
    size_change = round((( (value-smin) /smax)*2) - 1,2)
    final_size = (size_change*max_size_change) + 1
    return final_size*120
    
    


# SINCE A SINGLE FUNCTION CANNOT UPDATE MULTIPLE OUTPUTS AS BEFORE...
# ploty?4970

@app.callback(Output('related-sentiment', 'children'),
              [Input(component_id='sentiment_term', component_property='value')],
              events=[Event('related-update', 'interval')])
def update_related_terms(sentiment_term):
    try:

        # get that set data from cache
        for i in range(100):
            related_terms = cache.get('related_terms', sentiment_term) # term: {mean sentiment, count}
            if related_terms:
                break
            time.sleep(0.1)

        if not related_terms:
            return None

        buttons = [html.Button('{}({})'.format(term, related_terms[term][1]), id='related_term_button', value=term, className='btn', type='submit', style={'background-color':'#4CBFE1',
                                                                                                                                                           'margin-right':'5px',
                                                                                                                                                           'margin-top':'5px'}) for term in related_terms]
        #size: related_terms[term][1], sentiment related_terms[term][0]
        

        sizes = [related_terms[term][1] for term in related_terms]
        smin = min(sizes)
        smax = max(sizes) - smin  

        buttons = [html.H5('Terms related to "{}": '.format(sentiment_term), style={'color':app_color_scheme['text']})]+[html.Span(term, style={'color':sent_color_scheme[round(related_terms[term][0]*2)/2],
                                                              'margin-right':'15px',
                                                              'margin-top':'15px',
                                                              'font-size':'{}%'.format(generate_size(related_terms[term][1], smin, smax))}) for term in related_terms]


        return buttons
        
    # log the possible errors
    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


#recent-trending div. updates time to time.
# term: [sent, size]

@app.callback(Output('recent-trending', 'children'),
              [Input(component_id='sentiment_term', component_property='value')],
              events=[Event('related-update', 'interval')])
def update_recent_trending(sentiment_term):
    try:
        query = """SELECT value FROM misc WHERE key = 'trending'"""
        c = conn.cursor()

        result = c.execute(query).fetchone()
        related_terms = pickle.loads(result[0])


# Redundant HTML for now
##        buttons = [html.Button('{}({})'.format(term, related_terms[term][1]), id='related_term_button', value=term, className='btn', type='submit', style={'background-color':'#4CBFE1',
##                                                                                                                                                           'margin-right':'5px',
##                                                                                                                                                           'margin-top':'5px'}) for term in related_terms]
        #size: related_terms[term][1], sentiment related_terms[term][0]
        

        sizes = [related_terms[term][1] for term in related_terms]
        smin = min(sizes)
        smax = max(sizes) - smin  

        buttons = [html.H5('Recently Trending Terms: ', style={'color':app_color_scheme['text']})]+[html.Span(term, style={'color':sent_color_scheme[round(related_terms[term][0]*2)/2],
                                                              'margin-right':'15px',
                                                              'margin-top':'15px',
                                                              'font-size':'{}%'.format(generate_size(related_terms[term][1], smin, smax))}) for term in related_terms]


        return buttons
        

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


            
            

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})


external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_js:
    app.scripts.append_script({'external_url': js})

server = app.server
dev_server = app.run_server
