import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, title='Contact', name='Contact', order=10)

green_text = {'color':'blue'}

def layout():
    return dbc.Row([
        html.Hr(),
        dbc.Col([
            html.Br(),
                    html.A([
                        html.Img(src='assets/myImg.jpg', 
                                    alt='Image Alt Text', 
                                    style={'width': '300px', 'height': '350px'},
                                    ), 
                    ], 
                    href= ''),
            html.Br(),
]),
        dbc.Col([
    dcc.Markdown('# Abdel Kader A Geraldo', className='mt-3'), 
    dcc.Markdown('### Personal info', style={'color':'gray'}), 
    dcc.Markdown('Email', style=green_text),
    dcc.Markdown('ageraldo@umass.edu'),
    dcc.Markdown('Linkedin', style=green_text),
    dcc.Markdown('[https://www.linkedin.com/in/aka-gera/](https://www.linkedin.com/in/aka-gera/)', link_target='_blank'),
    html.Br(),


   ], width={'size':7, 'offset':0}),
   html.Br(),
   html.Hr(),
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
], justify='center')