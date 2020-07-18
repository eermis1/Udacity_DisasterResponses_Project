import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


def return_graphs():

    engine = create_engine('sqlite:///../data/CleanDataDB.db')
    df = pd.read_sql_table('clean_dataset', engine)

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df["message_length"] = df["message"].str.len()
    df_direct = df[df["genre"] == "direct"]
    df_social = df[df["genre"] == "social"]
    df_news = df[df["genre"] == "news"]


    print(df.head())
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    text= genre_counts,
                    textposition="outside",
                    marker_color='darkblue',
                    marker_line_color='black',
                    marker_line_width=1.5, opacity=0.65
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },


        # 2nd Graph - Average Message Length of Genres
        {
            'data': [
                Bar(
                    x=df.groupby("genre")["message_length"].mean().index,
                    y=df.groupby("genre")["message_length"].mean().values.round(1),
                    text=df.groupby("genre")["message_length"].mean().values.round(1),
                    textposition="outside",
                    marker_color='darkblue',
                    marker_line_color='black',
                    marker_line_width=1.5, opacity=0.65
                )
            ],

            'layout': {
                'title': {
                    'text': 'Average Message Length of Genres',
                    'x': '0.5',
                    'y': '0.9'
                },
                'yaxis': {
                    'title': "Average of Message Length"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # 3rd Graph - Message Length per ID
        {
            'data': [
                Bar(
                    x=df_direct["id"],
                    y=df_direct["message_length"],
                    marker_color="green",
                    name="Genre-Direct"
                ),
                Bar(
                    x=df_news["id"],
                    y=df_news["message_length"],
                    marker_color="indianred",
                    name="Genre-News"
                ),
                Bar(
                    x=df_social["id"],
                    y=df_social["message_length"],
                    marker_color="blue",
                    name="Genre-Social"
                )
            ],

            'layout': {
                'title': {
                    'text': 'Message Length per ID',
                    'x': '0.5',
                    'y': '0.9'
                },
                'yaxis': {
                    'title': "Message Length"
                },
                'xaxis': {
                    'title': "ID"
                }
            }
        },

        # 4th Graph - Count of Categories
        {
            'data': [
                Bar(
                    x= df.iloc[:, 5:-1].columns,
                    y= df.iloc[:, 5:-1].sum().values,
                    text= df.iloc[:, 5:-1].sum().values,
                    textposition="outside",
                    marker_color='darkblue',
                    marker_line_color='black',
                    marker_line_width=1.5, opacity=0.65
                )
            ],

            'layout': {
                'title': {
                    'text': 'Count of Categories',
                    'x': '0.5',
                    'y': '0.9'
                },
                'yaxis': {
                    'title': "Count"
                }
            }
        }
    ]

    return graphs