import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', order=0)

# resume sample template from https://zety.com/

professional_text = '''
I am a highly experienced and dedicated applied mathematician with a Ph.D. and
over six years of experience in innovative research. My contributions to the discipline
have been acknowledged with prominent awards, such as the Distinct Thesis Award
and the Don Catlin Award for Outstanding Achievement in Applied and Computational
Mathematics. I have a strong track record of publishing multiple research publications,
offering vital insights to the scientific community. My expertise lies in applying
mathematical principles and scientific computational software to solve challenging
real-world problems, particularly in molecular dynamics simulation. I am thrilled to 
continue making significant contributions to the advancement of knowledge and innovation.
'''

image_1 = 'assets/fig_neld_inv/pef_Density.png'
image_2 = 'assets/fig_neld_inv/shear_Density.png'
image_pub = 'https://www.devicealliance.org/wp-content/uploads/2020/06/Publication-icon.png'
image_contact = 'assets/contacts.png'
LinkStyle = {
    'textDecoration': 'none',  
    'width' : 3,
    'text-align-last' : 'center' ,
    'margin': 'auto',
}
ImageStyle={'width': '300px', 'height': '250px'}
layout = html.Div([
    html.Br(),
    dcc.Markdown('# Abdel Kader A Geraldo, Ph.D.', style={'textAlign':'center'}),
    html.Br(), 

    dcc.Markdown('### Professional Summary', style={'textAlign': 'center'}),
    html.Hr(),
    dcc.Markdown(professional_text, 
             style={'textAlign': 'center', 
                    'whiteSpace': 'pre',
                    'fontSize': 18}
                    ),
    html.Br(),
    html.Br(),
    dbc.Col(
    [ 
        html.Hr(), 
        dbc.Row(
            [
                dbc.Col(
                    [
                                    
                    html.A([
                        html.Img(src=image_1, 
                                    alt='Image Alt Text', 
                                    style=ImageStyle
                                    ),
                        html.Br(),
                        html.H3(['Developed Applications'],
                                    style={ 
                                        'textDecoration': 'none', 
                                        'text-align-last' : 'center' ,
                                        'margin': 'auto',  
                                        }
                        ),
                    ], 
                    href='myapps'),
                    ],
                        style=LinkStyle,
                ),#####################################
                dbc.Col(
                    [
                                    
                    html.A([
                        html.Img(src=image_2, 
                                    alt='Image Alt Text', 
                                    style=ImageStyle
                                    ),
                        html.Br(),
                        html.H3(['Research Interests'],
                                    style={ 
                                        'textDecoration': 'none', 
                                        'text-align-last' : 'center' ,
                                        'margin': 'auto',  
                                        }
                        ),
                    ], 
                    href='researchinterest'),
                    ],
                        style=LinkStyle,
                ),#####################################
            ]
        ) 
    ] 
    ),
    html.Br(),#################### COLUMN1 ###############################
    dbc.Col(
    [ 
        html.Hr(), 
        dbc.Row(
            [
                dbc.Col(
                    [
                                    
                    html.A([
                        html.Img(src=image_pub, 
                                    alt='Image Alt Text', 
                                    style=ImageStyle
                                    ),
                        html.Br(),
                        html.H3(['Publications'],
                                    style={ 
                                        'textDecoration': 'none', 
                                        'text-align-last' : 'center' ,
                                        'margin': 'auto',  
                                        }
                        ),
                    ], 
                                    href='publications'),
                    ],
                        style=LinkStyle,
                ),#####################################
                dbc.Col(
                    [
                                    
                    html.A([
                        html.Img(src=image_contact, 
                                    alt='Image Alt Text', 
                                    style=ImageStyle
                                    ),
                        html.Br(),
                        html.H3(['Contact Information'],
                                    style={ 
                                        'textDecoration': 'none', 
                                        'text-align-last' : 'center' ,
                                        'margin': 'auto',  
                                        }
                        ),
                    ], 
                    href='contact'),
                    ],
                        style=LinkStyle,
                ),#####################################
            ]
        ) 
    ]
    ),    
    html.Br(),#################### COLUMN2 ###############################
    
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
