import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from .side_bar import sidebar

dash.register_page(__name__, title='Applications', name='Applications',order=1)
 
image_1 = 'assets/fig_neld_inv/pef_Density.png'
image_2 = 'assets/fig_neld_inv/shear_Density.png'
image_data = 'assets/dataAnalyst.png'
image_nav = 'assets/nav.png'
image_apps = 'assets/apps.png'


page_1 = 'app4'
def layout():
    return html.Div([
    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar()
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
                    html.H1('Developed Applications', style={'textAlign':'center'}), 
                    html.Hr(), 
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                                
                                html.A([
                                    html.Br(),
                                    html.Img(src=image_1, 
                                                alt='Image Alt Text', 
                                                style={'width': '200px', 'height': '200px'}),
                                    html.H3(['NELD Limit Cycle'],
                                                style={ 
                                                    'text-align-last' : 'center' ,
                                                    'margin': 'auto',   
                                                    'color': 'grey'}
                                    ),
                                ], 
                                                href='app4'),
                                ],
                                  style={
                                    'width' : 3,
                                    'text-align-last' : 'center' ,
                                    'margin': 'auto',   
                                    'color': 'grey'}
                            ),##################################### 
                            # dbc.Col([            
                            #     html.A([

                            #         html.Br(), 
                            #         html.Img(src=image_nav, 
                            #                     alt='Image Alt Text', 
                            #                     style={'width': '200px', 'height': '200px'}),
                            #         html.H3(['Navigation'],
                            #         style={ 
                            #             'text-align-last' : 'center' ,
                            #             'margin': 'auto',   
                            #             'color': 'grey'}
                            #         ),
                            #         ],
                            #     href='app3'),
                            #     ],
                            #       style={
                            #         'width' : 3,
                            #         'text-align-last' : 'center' ,
                            #         'margin': 'auto',   
                            #         'color': 'grey'}
                            # ),#####################################
                            # dbc.Col([            
                            #     html.A([

                            #         html.Br(), 
                            #         html.Img(src=image_apps, 
                            #                     alt='Image Alt Text', 
                            #                     style={'width': '200px', 'height': '200px'}),
                            #         html.H3(['Bolza Example Approximation'],
                            #         style={ 
                            #             'text-align-last' : 'center' ,
                            #             'margin': 'auto',   
                            #             'color': 'grey'}
                            #         ),
                            #         ],
                            #     href='app5'),
                            #     ],
                            #       style={
                            #         'width' : 3,
                            #         'text-align-last' : 'center' ,
                            #         'margin': 'auto',   
                            #         'color': 'grey'}
                            # ),##################################### 

                        ]
                    ),
                    ########################## ROW 2 ##############
                html.Hr(), 

                    # dbc.Row(
                    #     [
 
                    #         dbc.Col([            
                    #             html.A([ 
                    #                 html.Br(),
                    #                 html.Img(src=image_data, 
                    #                             alt='Image Alt Text', 
                    #                             style={'width': '200px', 'height': '200px'}),
                    #                 html.H3(['Supervised Learning Classification'],
                    #               style={ 
                    #                 'text-align-last' : 'center' ,
                    #                 'margin': 'auto',   
                    #                 'color': 'grey'}
                    #                 ),
                    #             ], 
                    #             href='app2'),
                    #             ],
                    #               style={
                    #                 'width' : 3,
                    #                 'text-align-last' : 'center' ,
                    #                 'margin': 'auto',   
                    #                 'color': 'grey'}
                    #         ),#####################################  
                    #         dbc.Col([            
                    #             html.A([

                    #                 html.Br(), 
                    #                 html.Img(src=image_data, 
                    #                             alt='Image Alt Text', 
                    #                             style={'width': '200px', 'height': '200px'}),
                    #                 html.H3(['Deep Neural Network Classification'],
                    #                 style={ 
                    #                     'text-align-last' : 'center' ,
                    #                     'margin': 'auto',   
                    #                     'color': 'grey'}
                    #                 ),
                    #                 ],
                    #             href='app6'),
                    #             ],
                    #               style={
                    #                 'width' : 3,
                    #                 'text-align-last' : 'center' ,
                    #                 'margin': 'auto',   
                    #                 'color': 'grey'}
                    #         ),

                    #     ]
                    # )
              
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    )
])
           