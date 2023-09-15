import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, order=3)

green_text = {'color':'blue'}

def layout():
    return dbc.Row([
        dbc.Col([
    dcc.Markdown('# Abdel Kader A Geraldo', className='mt-3'), 
    dcc.Markdown('### Personal info', style={'color':'gray'}), 
    dcc.Markdown('Email', style=green_text),
    dcc.Markdown('ageraldo@umass.edu'),
    dcc.Markdown('Linkedin', style=green_text),
    dcc.Markdown('[https://www.linkedin.com/in/aka-gera/](https://www.linkedin.com/in/aka-gera/)', link_target='_blank'),
   ], width={'size':6, 'offset':2})
], justify='center')