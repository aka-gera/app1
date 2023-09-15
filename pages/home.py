import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', order=0)

# resume sample template from https://zety.com/
layout = html.Div([
    dcc.Markdown('# Abdel Kader A Geraldo', style={'textAlign':'center'}), 

    dcc.Markdown('### Professional Summary', style={'textAlign': 'center'}),
    html.Hr(),
    dcc.Markdown('I am a highly experienced and dedicated applied mathematician with a Ph.D. and\n', 
                 'over six years of experience in innovative research. My contributions to the discipline\n',
                  'have been acknowledged with prominent awards, such as the Distinct Thesis Award\n',
                   'and the Don Catlin Award for Outstanding Achievement in Applied and Computational\n',
                    'Mathematics. I have a strong track record of publishing multiple research publications,\n'
                     'offering vital insights to the scientific community. My expertise lies in applying\n'
                      'mathematical principles and scientific computational software to solve challenging\n'
                       'real-world problems, particularly in molecular dynamics simulation. I am thrilled to \n',
                       'continue making significant contributions to the advancement of knowledge and innovation,\n',
                         style={'textAlign': 'center', 'white-space': 'pre'}),

    dcc.Markdown('### Skills', style={'textAlign': 'center'}),
    html.Hr(),
  
 
])
