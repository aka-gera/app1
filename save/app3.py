import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from ..pages.side_bar import sidebar
# Import required libraries
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash import Dash, dcc, html,  Input, Output,callback
from dash.dependencies import Input, Output 
from dash import no_update
import dash_bootstrap_components as dbc


import pyproj

dash.register_page(__name__, title='Navigation', name='Navigation',order=10)





shw = 0

dropdown_options_style = {'color': 'white', 'background-color' : 'gray'}

dropdown_options = [
    {'label': 'All Features', 'value': 'ALL', 'style': dropdown_options_style}
]

for col in range(100):
    dropdown_options.append({'label': 'Feature'+str(col), 'value': 'Feature'+str(col), 'style':  dropdown_options_style})

box_style={
            'width':'80%',
            'padding':'3px',
            'font-size': '20px',
            'text-align-last' : 'center' ,
            'margin': 'auto',  # Center-align the dropdown horizontally
            'background-color' : 'black',
            'color': 'black'
            }


def layout():
    return  html.Div(
    style={
        'color' : 'black',
        # 'backgroundColor': '',  # Set the background color of the app here
        'height': '100vh'  # Set the height of the app to fill the viewport
    },
    children=[
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1('Coordinates Finder',
            style={'textAlign': 'center',
                   'color': 'white',
                   'background-color' : 'black',
                   'font-size': 40
                   }
            ),
    html.Br(),



    html.Br(),
    html.Br(),


 
    html.Br(),
    #       # Create an outer division
    #  html.Div([
    #         html.Div([
    #             html.Div([
    #               dcc.Dropdown(
    #                   id='site-dropdown1',
    #                   options=dropdown_options,
    #                   value='ALL',
    #                   placeholder='Select a feature',
    #                   style=box_style,
    #                   searchable=True
    #               ) ,
    #               html.Div(id='output-graph1') ,
    #             ]),
    #     ]),
    #     html.Div([
    #         dcc.Dropdown(
    #             id='site-dropdown2',
    #             options=[
    #                     {'label':  'Logistic Regression',          'value': 'LG',   'style':  dropdown_options_style},
    #                     {'label': 'Decision Tree Classifier',      'value': 'DT',   'style':  dropdown_options_style},
    #                     {'label': 'K-Nearest Neighbors',           'value': 'KNN',  'style':  dropdown_options_style},
    #                     {'label': 'Support Vector Classification', 'value': 'SVC',  'style':  dropdown_options_style},
    #                     {'label': 'Gaussian Naive Bayes',          'value': 'NB',   'style':  dropdown_options_style},
    #                     {'label': 'Stochastic Gradient Descent',   'value': 'SGD',  'style':  dropdown_options_style}
    #                     ],
    #             value='LG',
    #             placeholder='Select a Machine Learning Classifier',
    #             style=box_style,
    #             searchable=True,
    #         ) ,
    #     ]),
    #  ]),

####################################################################################

    html.Div([
    html.H1("Enter the coordinates of your location",
            style={'textAlign': 'center',
                        'color': 'white',
                        'background-color' : 'black',
                        'font-size': 20
                        }
             ),
    html.Div([
        html.Label('X1 : '),   
        dcc.Input(
            id='input-x1',
            type='number',
            value=0,  # Initial value
            debounce=True   
        ),
        html.Label('Y1 : '),   
        dcc.Input(
            id='input-y1',
            type='number',
            value=0,  # Initial value
            debounce=True  # Delay the callback until typing stops
        ),
    ]), 
############################
    html.Div([
        html.Label('X2 : '),   
        dcc.Input(
            id='input-x2',
            type='number',
            value=0,  # Initial value
            debounce=True   
        ),
        html.Label('Y2 : '),   
        dcc.Input(
            id='input-y2',
            type='number',
            value=0,  # Initial value
            debounce=True  # Delay the callback until typing stops
        ),
        # html.Div(id='output-v2')  # Placeholder for displaying output
    ]), 
#############################
    html.Div([
        html.Label('X3 : '),   
        dcc.Input(
            id='input-x3',
            type='number',
            value=0,  # Initial value
            debounce=True   
        ),
        html.Label('Y3 : '),   
        dcc.Input(
            id='input-y3',
            type='number',
            value=0,  # Initial value
            debounce=True  # Delay the callback until typing stops
        ),
    ]), 
############################
    html.Div([
        html.Label('X4 : '),   
        dcc.Input(
            id='input-x4',
            type='number',
            value=0,  # Initial value
            debounce=True   
        ),
        html.Label('Y4 : '),   
        dcc.Input(
            id='input-y4',
            type='number',
            value=0,  # Initial value
            debounce=True  # Delay the callback until typing stops
        ),
        # html.Div(id='output-v2')  # Placeholder for displaying output
    ]), 
#############################
],
             style={'textAlign': 'center',
                        'color': 'white',
                        'background-color' : 'black',
                        'font-size': 20
                        }
             ),
####################################################################################
html.Br(),
html.Br(),
    html.Div([
    html.H1("Entered Geographic Coordinates",
            style={'textAlign': 'center',
                        'color': 'white',
                        'background-color' : 'black',
                        'font-size': 20
                        }
             ),
  
dbc.Container(
    [
         dbc.Card(
        dbc.Row(
            [
                dbc.Col(html.Div('Longitude'), width=0),
                dbc.Col(html.Div('Latitude'), width=0),
            ], 
              style={
                            'textAlign': 'center',
                            # 'color': 'grey',
                            # 'background-color': 'black',
                            'font-size': 25
                        }
        ),          
            body=True,  #  
            color="grey",  #             
            style={
                'width': '75%',  # Set the width of the card to 75% of the page
                'margin': '0 auto'  # Center the card horizontally
            }
        ),
    dbc.Card(
        dbc.Row(
            [
                dbc.Col(html.Div(id='output-v1'), width=12,),
                dbc.Col(html.Div(id='output-v2'), width=12),
                dbc.Col(html.Div(id='output-v3'), width=12),
                dbc.Col(html.Div(id='output-v4'), width=12),
            ]
        ),
        body=False,  #  
            color="grey",  #             
            style={
                'width': '75%',  # Set the width of the card to 75% of the page
                'margin': '0 auto'  # Center the card horizontally
            }
        ),
    ],
    fluid=True,
    style={'textAlign': 'center',
        'color': 'white',
        'background-color' : 'black',
        'font-size': 20
        }
), 
],
             style={'textAlign': 'center',
                        'color': 'white',
                        'background-color' : 'black',
                        'font-size': 20
                        }
             ),

        html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        html.Div( ['Click the icon below to view the position on google map']
    ),
        html.Div(id='output-txt')
    ],
        style={'textAlign': 'center', 'color': 'white', 'background-color' : 'black',   'font-size': 20},
        ),


 
    html.Br(),
    html.Br(),
    html.Div([
        html.A(
            html.Img(src='https://img.icons8.com/color/48/000000/github.png'),
            href='https://github.com/aka-gera',
            target='_blank'
        ),
        html.A(
            html.Img(src='https://img.icons8.com/color/48/000000/linkedin.png'),
            href='https://www.linkedin.com/in/aka-gera/',
            target='_blank'
        ),
        html.A(
            html.Img(src='https://img.icons8.com/color/48/000000/youtube.png'),
            href='https://www.youtube.com/@aka-Gera',
            target='_blank'
        ),
    ], style={'display': 'flex', 'justify-content': 'center'})


])








@callback([
        Output('output-txt', 'children'),
        Output('output-v1', 'children'),
        Output('output-v2', 'children'),
        Output('output-v3', 'children'),
        Output('output-v4', 'children'),
    ],
    [
        # Input('site-dropdown1', 'value'),
        # Input('site-dropdown2', 'value'),
        Input('input-x1', 'value'),
        Input('input-y1', 'value'),
        Input('input-x2', 'value'),
        Input('input-y2', 'value'),
        Input('input-x3', 'value'),
        Input('input-y3', 'value'),
        Input('input-x4', 'value'),
        Input('input-y4', 'value'),
        ],
        prevent_initial_call=True
              )



def update_output(input_x1,input_y1,
                  input_x2,input_y2,
                  input_x3,input_y3,
                  input_x4,input_y4):
    # Define the coordinate transformation
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:32631")
    transformer = pyproj.Transformer.from_crs(web_mercator, wgs84) 


    try:
        ##############################1
        input_lat, input_long = transformer.transform(input_x1, input_y1) 
        v1 = dbc.Container( 
                        dbc.Row(
                            [
                                dbc.Col(html.Div(f'{input_long:.7f}'), width=6),
                                dbc.Col(html.Div(f'{input_lat:.7f}'), width=6),
                            ]
                        ), 
                        style={
                            'textAlign': 'center',
                            'color': 'grey',
                            'background-color': 'black',
                            'font-size': 20
                        }
        )
        txt_output1 =html.A(
                        html.Img(src="https://img.icons8.com/color/48/000000/map.png"), 
                        href=f'https://www.google.com/maps/search/{input_lat},{input_long}?sa=X&ved=2ahUKEwiN9cXmjsKBAxXWM1kFHYGUDQUQ8gF6BAgQEAA&ved=2ahUKEwiN9cXmjsKBAxXWM1kFHYGUDQUQ8gF6BAgREAI',
                        target='_blank'
        )
        ##############################2
        input_lat2, input_long2 = transformer.transform(input_x2, input_y2) 
        v2 = dbc.Container( 
                        dbc.Row(
                            [
                                dbc.Col(html.Div(f'{input_long2:.7f}'), width=6),
                                dbc.Col(html.Div(f'{input_lat2:.7f}'), width=6),
                            ]
                        ), 
                        style={
                            'textAlign': 'center',
                            'color': 'grey',
                            'background-color': 'black',
                            'font-size': 20
                        }
        )
        txt_output2 =html.A(
                        html.Img(src="https://img.icons8.com/color/48/000000/map.png"),
                        href =f'https://www.google.com/maps/dir/{input_lat},{input_long}/{input_lat2},{input_long2}/@{input_lat},{input_long},14.98z?entry=ttu',
                        target='_blank'
        )
        ################################3
        input_lat3, input_long3 = transformer.transform(input_x3, input_y3) 
        v3 = dbc.Container( 
                        dbc.Row(
                            [
                                dbc.Col(html.Div(f'{input_long3:.7f}'), width=6),
                                dbc.Col(html.Div(f'{input_lat3:.7f}'), width=6),
                            ]
                        ), 
                        style={
                            'textAlign': 'center',
                            'color': 'grey',
                            'background-color': 'black',
                            'font-size': 20
                        }
        )
        txt_output3 =html.A(
                        html.Img(src="https://img.icons8.com/color/48/000000/map.png"),
                        href =f'https://www.google.com/maps/dir/{input_lat},{input_long}/{input_lat2},{input_long2}/{input_lat3},{input_long3}/{input_lat},{input_long}/@{input_lat},{input_long},14.98z?entry=ttu',
                        target='_blank'
        )
        #################################4
        input_lat4, input_long4 = transformer.transform(input_x4, input_y4) 
        v4 = dbc.Container( 
                        dbc.Row(
                            [
                                dbc.Col(html.Div(f'{input_long4:.7f}'), width=6),
                                dbc.Col(html.Div(f'{input_lat4:.7f}'), width=6),
                            ]
                        ), 
                        style={
                            'textAlign': 'center',
                            'color': 'grey',
                            'background-color': 'black',
                            'font-size': 20
                        }
        )
        txt_output4 =html.A(
                        html.Img(src="https://img.icons8.com/color/48/000000/map.png"),
                        href =f'https://www.google.com/maps/dir/{input_lat},{input_long}/{input_lat2},{input_long2}/{input_lat3},{input_long3}/{input_lat4},{input_long4}/{input_lat},{input_long}/@{input_lat},{input_long},14.98z?entry=ttu',
                        target='_blank'
        )
    except Exception as e:
        v1 = f"Error: {str(e)}"
        v2 = v3 = v4 = ""

    if (input_lat2 == 0 or input_long2 == 0) and (input_lat3 == 0 or input_long3 == 0) and (input_lat4 == 0 or input_long4 == 0):
        txt_output = txt_output1
    elif (input_lat3 == 0 or input_long3 == 0) and (input_lat4 == 0 or input_long4 == 0):
        txt_output = txt_output2
    elif (input_lat4 == 0 or input_long4 == 0):
        txt_output = txt_output3
    else:
        txt_output = txt_output4

    return [txt_output, v1, v2, v3, v4]
