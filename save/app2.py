 

# -*- coding: utf-8 -*-
"""v5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jbkr8M9kFa_idP2gDYr0JYim5aLt-HNz
"""

# !pip install dash
# # !pip install dash==1.19.0
# !pip install jupyter_dash
# !pip install --upgrade plotly
# !pip install dash --upgrade
# !pip install dash_bootstrap_components

"""<!--  -->"""

# Import required libraries
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash import Dash, dcc, html, dash_table, Input, Output, callback
from dash.dependencies import Input, Output   
from dash import no_update 
 
# from sklearn.model_selection import train_test_split  
  

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix 

from my_dash_class.my_files import dash_deco,plot_dash  
from my_dash_class.my_data import getData, DashToDataFrame,download
from my_dash_class.my_learning_alg import SuperLearning


barName = 'Supervised Learning Classification'

dash.register_page(__name__, title=barName, name=barName,order=9)
 

# call class
dash_deco = dash_deco()
DashToDataFrame = DashToDataFrame()
download = download() 
 

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
    return html.Div(
    style=dash_deco.app_style,
    children=[
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1('Dataset Analysis Via Supervised Learning Algorithms',
            style={'textAlign': 'center',
                   'color': 'white',
                   'background-color' : 'black',
                   'font-size': 40
                   }
            ),
    html.Br(),

    html.Br(),
    html.Div([
        html.H1("Upload the training data",
            style={'textAlign': 'center',
                        'color': 'white',
                        'background-color' : 'black',
                        'margin': 'auto',  # Center-align the dropdown horizontally
                        'font-size': 20
                        }
             ),
    dcc.Upload(
         id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File with extension .data, .csv, or .xls')
        ]),
        style={
            'display': 'flex',
            'justify-content': 'center',
            'width': '50%',
            'height': '50px',
            'margin': 'auto',  # Center-align the dropdown horizontally
            'color': 'black',
            'background-color' : 'grey',
            } ,
        multiple=True
    ), 
]),

    html.Br(),
    html.Div([
        html.H1("Upload the data to predict",
            style={'textAlign': 'center',
                        'color': 'white',
                        'background-color' : 'black',
                        'margin': 'auto',  # Center-align the dropdown horizontally
                        'font-size': 20
                        }
             ),
    dcc.Upload(
      id='upload-data2',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File with extension .data, .csv, or .xls')
        ]),
        style={
            'display': 'flex',
            'justify-content': 'center',
            'width': '50%',
            'height': '50px',
            'margin': 'auto',  # Center-align the dropdown horizontally
            'color': 'black',
            'background-color' : 'grey',
            } ,
        multiple=True
    ), 
]),

    html.Br(),
    html.Br(),



    html.Div('Analysis of the dataset and evaluation of various classification machine algorithms',
            style={'textAlign': 'center',
                'color': 'white',
                'background-color' : 'black',
                'font-size': 35
                }
            ),
    html.Br(),
          # Create an outer division
     html.Div([
            html.Div([
                html.Div([
                  dcc.Dropdown(
                      id='site-dropdown1',
                      options=dropdown_options,
                      value='ALL',
                      placeholder='Select a feature',
                      style=box_style,
                      searchable=True
                  ) ,
                  html.Div(id='output-graph1') ,
                ]),
        ]),
        html.Div([
            dcc.Dropdown(
                id='site-dropdown2',
                options=[
                        {'label':  'Logistic Regression',          'value': 'LG',   'style':  dropdown_options_style},
                        {'label': 'Decision Tree Classifier',      'value': 'DT',   'style':  dropdown_options_style},
                        {'label': 'K-Nearest Neighbors',           'value': 'KNN',  'style':  dropdown_options_style},
                        {'label': 'Support Vector Classification', 'value': 'SVC',  'style':  dropdown_options_style},
                        {'label': 'Gaussian Naive Bayes',          'value': 'NB',   'style':  dropdown_options_style},
                        {'label': 'Stochastic Gradient Descent',   'value': 'SGD',  'style':  dropdown_options_style}
                        ],
                value='LG',
                placeholder='Select a Machine Learning Classifier',
                style=box_style,
                searchable=True,
            ) ,
        html.Div([
            html.Div(id='output-graph2', style={'width': '50%', 'display': 'inline-block'}),
            html.Div(id='output-graph3', style={'width': '50%', 'display': 'inline-block'}),
        ]),
        ]),
     ]),
        html.Br(),
    html.Br(),
    html.Div(id='output-text'),
    html.Br(),
    html.Br(),
 html.Div([
    html.H1("Download the predicted result based on the selected machine learning algorithm: ",
            style={'textAlign': 'center',
                        'color': 'white',
                        'background-color' : 'black',
                        'font-size': 20
                        }
             ),
    dcc.Download(id="download-button"),
    html.Button("Download Prediction",
                id="btn-download",
                style=dash_deco.default_style_buttom
                ),
        ],
    style={
        'display': 'flex',
        'justify-content': 'center',
        }
          ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([dash_deco.signature]),
    html.Br(),


])








@callback([
        Output('output-graph1', 'children'),
        Output('output-graph2', 'children'),
        Output('output-graph3', 'children'),
        Output('output-text', 'children'),
         Output("download-button", "data"),
    ],
    [
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),


        Input('upload-data2', 'contents'),
        Input('upload-data2', 'filename'),
        Input('site-dropdown1', 'value'),
        Input('site-dropdown2', 'value'),
        Input('btn-download', 'n_clicks'),
        ],
        prevent_initial_call=True
              )



def update_output(list_of_contents, list_of_names,list_of_contents2, list_of_names2,feature,ml,n_clicks):
    if [list_of_contents,list_of_contents2] is not None: 
        
        df = DashToDataFrame.dash_to_df(list_of_contents, list_of_names)
        dfpred = DashToDataFrame.dash_to_df(list_of_contents2, list_of_names2)
 

        getDat = getData(df,dfpred) 
        dff,X_train, X_test, y_train, y_test,X_pred, cmLabel,typOfVar,mapping = getDat.Algorithm() 
        filtered_df = dff[dff.columns[0:-1]]



        plot_d = plot_dash(filtered_df)

        if feature == 'ALL':
            figure1 =  dcc.Graph( figure = plot_d.plot_history_all_dash() )
        else:
            figure1 =  dcc.Graph( figure = plot_d.plot_history_dash(feature) )


        ML = SuperLearning(X_train, X_test, y_train, y_test,X_pred) 

        y_pred,y_predpred,scre = ML.ChooseML(ml) 
        fig2 = dcc.Graph( figure = plot_d.plot_confusion_matrix_dash(y_test,y_pred,cmLabel,shw))
        fig3 = dcc.Graph( figure = plot_d.plot_classification_report_dash(y_test,y_pred,cmLabel,shw))
  

        txt_output = html.Div( ['The overall accuracy of the selected algorithm is ',f'{scre*100 :.2f}','%'],
            style=dash_deco.default_style
                )

        if df.shape[1] in typOfVar:
            y_predpred = pd.DataFrame(y_predpred).replace(mapping).values

        if n_clicks is None:
            butpred = dash.no_update   
        else:
            butpred =download.dfDownload(y_predpred)

        return  [  figure1,fig2,fig3,txt_output,butpred]
 