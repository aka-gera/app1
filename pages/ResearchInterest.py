import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from .side_bar import sidebar

dash.register_page(__name__, title='Research Interests', name='Research Interests', order=2)

def layout():
    return html.Div([
  html.Br(),  
    dcc.Markdown('# Research Interests', style={'textAlign': 'center'}),
 html.Br(),  
 html.Hr(), 
        dbc.Row([
        dbc.Col([
            dcc.Markdown('#### MOLECULAR DYNAMICS PERIODIC BOUNDARY CONDITIONS (PBCS)'),
            dcc.Markdown('THREE‑DIMENSIONAL ALGORITHM')
        ], width=2),
        dbc.Col([
            dcc.Markdown('Developed an improved Three‑Dimensional automorphism remapping PBC algorithm'
                         ' for simulating molecular dynamics, with particles exhibiting an average flow ' 
                         ' consistent with a homogeneous background flow',
                         className='ms-3'),
            html.Hr(), 
            dcc.Markdown('Implemented my proposed PBC algorithm along with the existing PBCs in C++, '
                         'Matlab, and Python, and analyzed their performance ',
              className='ms-3'),
        ], width=5)
    ], justify='center'),
    html.Hr(),

###################################################################################################################
    dbc.Row([
        dbc.Col([
            dcc.Markdown('####  NONEQUILIBRIUM MOLECULAR DYNAMICS'),
            dcc.Markdown('CONVERGENCE ANALYSIS')
        ], width=2),
        dbc.Col([
            dcc.Markdown('Performed strong convergence analysis, both numerically and analytically,'
                         'of nonequilibrium molecular dynamics simulations using first and second‑order integrators',
                         className='ms-3'),
            html.Hr(), 
            dcc.Markdown('Utilized C++, Matlab, Julia, and Scilab to implement the integrators'
                         ' and conducted simulations of both equilibrium and nonequilibrium molecular dynamics',
              className='ms-3'),
            html.Hr(), 
            dcc.Markdown('Established analytically the exponential convergence to a periodic invariant'
                         ' measure of the nonequilibrium Langevin dynamics (NELD) under planar flow, '
                         'using the automorphism remapping and PBCs technique',
              className='ms-3'),
        ], width=5)
    ], justify='center'),
    html.Hr(),


        ###################################################################################################################
    dbc.Row([
        dbc.Col([
            dcc.Markdown('####  DATA SCIENCE'),
            dcc.Markdown('ALGORITHMS')
        ], width=2),
        dbc.Col([
            dcc.Markdown('Designed and deployed both supervised and unsupervised machine learning'
                         ' techniques for data analysis, while also providing guidance and supervision to undergraduate students',
                         className='ms-3'),
            html.Hr(), 
            dcc.Markdown('Utilized visualization libraries like Plotly, Seaborn,'
                         ' and Dashboards to effectively communicate data science results to diverse audiences',
              className='ms-3'),
            html.Hr(), 
            dcc.Markdown('Developed and deployed dashboard applications for performing data analysis,'
                         ' including classification and regression tasks, across various datasets',
              className='ms-3'),
        ], width=5)
    ], justify='center'),
    html.Hr(),


    ###################################################################################################################
    dbc.Row([
        dbc.Col([
            dcc.Markdown('####  SCIENTIFIC COMPUTATIONAL'),
            dcc.Markdown('ALGORITHMS')
        ], width=2),
        dbc.Col([
            dcc.Markdown('Designed and deployed machine learning neural network models for physics informatics,'
                         ' employing data-driven methodologies to investigate complex physical systems',
                         className='ms-3'),
            html.Hr(), 
            dcc.Markdown('Developed computational algorithms in Matlab for solving partial differential'
                         'equations (PDEs) using finite difference, finite element, or spectral methods',
              className='ms-3'),
            html.Hr(), 
            dcc.Markdown('Developed and programmed numerical schemes in Matlab and Python to'
                         'solve systems of ordinary differential equations',
              className='ms-3'),
        ], width=5)
    ], justify='center'),
    html.Hr(),
  
])