
import dash
from dash import dcc
from dash import html
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
import plotly.express as px
from dash import no_update
import dash_bootstrap_components as dbc 

import numpy as np
from numpy.core.multiarray import zeros

"""### 4th Order Hessian Approximation"""

# 4th order Hessian approximation from http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/#comment-5289
def Local(f,x,Coef,h,m,i,j,hi,hj) :

    Coef[i] = Coef[i] + hi*h
    Coef[j] = Coef[j] + hj*h
    return f(x, Coef, m)


def HESS1(f, x, aCoef, m):
    h = 1e-6
    n = len(aCoef)
    Hess = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            TF00 = Local(f,x,aCoef.copy(),h,m,i,j,1,-2)

            TF01 =  Local(f,x,aCoef.copy(),h,m,i,j,2,-1)

            TF02 = Local(f,x,aCoef.copy(),h,m,i,j,-2,1)

            TF03 = Local(f,x,aCoef.copy(),h,m,i,j,-1,2)

            TF0 = TF00 + TF01 + TF02 + TF03



            TF10 = Local(f,x,aCoef.copy(),h,m,i,j,-1,-2)

            TF11 =  Local(f,x,aCoef.copy(),h,m,i,j,-2,-1)

            TF12 = Local(f,x,aCoef.copy(),h,m,i,j,1,2)

            TF13 = Local(f,x,aCoef.copy(),h,m,i,j,2,1)

            TF1 = TF10 + TF11 + TF12 + TF13


            TF20 = Local(f,x,aCoef.copy(),h,m,i,j,2,-2)

            TF21 =  Local(f,x,aCoef.copy(),h,m,i,j,-2,2)

            TF22 = Local(f,x,aCoef.copy(),h,m,i,j,-2,-2)

            TF23 = Local(f,x,aCoef.copy(),h,m,i,j,2,2)

            TF2 = TF20 + TF21 - TF22 - TF23


            TF30 = Local(f,x,aCoef.copy(),h,m,i,j,-1,-1)

            TF31 =  Local(f,x,aCoef.copy(),h,m,i,j,1,1)

            TF32 = Local(f,x,aCoef.copy(),h,m,i,j,1,-1)

            TF33 = Local(f,x,aCoef.copy(),h,m,i,j,-1,1)

            TF3 = TF30 + TF31 - TF32 - TF33


            Hess[i, j] = (-63 * TF0 + 63 * TF1 + 44* TF2 + 74 * TF3) / (600 * h ** 2)

            Hess[j, i] = Hess[i, j]

    return Hess

"""### 4th Order Gradient Approximation"""

# 4th order gradian approximation from http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/#comment-5289
def Loc(f,x,Coef,h,m,i,hi) :

    Coef[i] = Coef[i] + hi*h
    return f(x, Coef, m)


def Diff1(f, x, aCoef,m):
  h=1e-6
  grad = np.zeros(len(aCoef))

  for i in range(len(aCoef)):
    grad[i] =((-Loc(f,x,aCoef.copy(),h,m,i,2) \
               + 8 * Loc(f,x,aCoef.copy(),h,m,i,1) \
               - 8 * Loc(f,x,aCoef.copy(),h,m,i,-1)  \
               + Loc(f,x,aCoef.copy(),h,m,i,-2)  ) / (12 * h))

  return grad

"""### Nonlinear Conjugate Gradian Method"""

def ConjugateGradient(Fun, x, y0, cas, m, tol):
  ff0 = -Diff1(Fun, x, y0,m)
  fff0 = HESS1(Fun, x, y0,m)
  s0 = ff0
  alpha=np.dot(ff0,s0)/np.dot(np.dot(s0,fff0),s0)
  y = y0 + alpha * s0

  k = 0 

  while max(abs(ff0))>tol :

    ff = -Diff1(Fun, x, y,m)
    fff = HESS1(Fun, x, y,m)
    s = ff

    if cas == 'FLETCHER-REEVES':
            beta=np.dot(ff,ff)/np.dot(ff0,ff0)
            beta=max([beta,0])
    elif cas == 'POLAK-RIIERE':
            beta=np.dot(ff,ff-ff0)/np.dot(ff0,ff0)
            beta=max([beta,0])
    elif cas == 'HESTENES-STIEFEL':
            beta=np.dot(ff,ff-ff0)/np.dot(ff-ff0,y0)
            beta=max([beta,0])



    s = s + beta*s0
    alpha=np.dot(ff,s)/np.dot(np.dot(s,fff),s)
    y = y0 + alpha*s
    y0=y
    s0=s
    k+=1
    ff0 = ff

    print('iter  ',k,'   grad ',max(abs(ff0)))
  return y

"""### Bolza Integral Function

### Main Cell
"""
 
def poly_pre_bolza1(x,aCoef,Inter):

  n = len(x)
  m=len(aCoef)
  X = np.tile(x,(1,m)).reshape(m,n).T

  Mp0 = np.tile(list(range(0,m)),(n,1))
  Mp1 = np.tile(list(range(1,m+1)),(n,1))

  y0 = X**Mp0
  y1 = X**Mp1

  M0 = np.zeros((n,1))
  M1 = np.zeros((n,1))

  for i in range(len(aCoef)):
    M0 +=aCoef[i]*y0[:,i:i+1]
    M1 +=aCoef[i]*y1[:,i:i+1]

  return (1-M0**2)**2 + M1**4


def Trapz(f,a,b):
  return (b-a)/(2*len(f))*(sum(2*f[1:-1]+f[0])+f[-1])


def poly_bolza(x,aCoef,Inter):
  f = poly_pre_bolza1(x,aCoef,Inter)
  I =Trapz(f,Inter['a'],Inter['b'])
  return I


def uFun(y,x):
  M = zeros(len(x))
  for i in range(len(y)):
    M += y[i]*x**i
  return M


  



tcouleur = 'plotly_dark'
bcouleur = 'navy'
fcouleur = 'white'
fsize = 20
   



def plot_fun(xx,yy,bheight,bwidth):

  fig = px.line(x=xx, y=yy, labels={'x':'x', 'y':'y'})
  
  fig.update_layout(
        barmode='overlay',
      # paper_bgcolor=bcouleur,  
      font=dict(color=fcouleur,size=fsize),  # Set the font color 
      title_x=0.5,
      title_y=0.95,
      template=tcouleur,
      autosize=False,
      height=bheight,
      width=bwidth, 
  )
  return fig









########################## APP BEGIN ###########################################################

 
dropdown_options_style = {'color': 'white', 'background-color' : 'gray'}

dropdown_options = [
    {'label': 'All Features', 'value': 'ALL', 'style': dropdown_options_style}
]


box_style={
            'width':'60%',
            'padding':'3px',
            'font-size': '20px',
            'text-align-last' : 'center' ,
            'margin': 'auto',  # Center-align the dropdown horizontally
            'background-color' : 'black',
            'color': 'black'
            } 

app =  dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
JupyterDash.infer_jupyter_proxy_config()

server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    style={
        'color' : 'black',
        'backgroundColor': 'black',  # Set the background color of the app here
        'height': '100vh'  # Set the height of the app to fill the viewport
    },
    children=[
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1('Polynomial Approximation for Bolza Example',
            style={'textAlign': 'center',
                   'color': 'white',
                   'background-color' : 'black',
                   'font-size': 40
                   }
            ),
    html.Br(),


    html.Br(),
    
    html.Hr(style={'border-color': 'white'}),
    html.Br(),

 
    html.Br(),
          # Create an outer division
     html.Div([ 
        html.Div([

        html.Div([
        html.H1("Enter the simulation parameters",
                style={'textAlign': 'center',
                            'color': 'grey',
                            'background-color' : 'black',
                            'font-size': 30
                            }
                ),
        html.Br(),
        html.Div([
            html.Label('Polynomial Degree__________: '),   
            dcc.Input(
                id='input-m',
                type='number',
                value=3,  # Initial value
                debounce=True   
            ),
        ]),
        html.Div([
            html.Label('Interpolation Point Number: '),   
            dcc.Input(
                id='input-n',
                type='number',
                value=4,  # Initial value
                debounce=True  # Delay the callback until typing stops
            ),
        ]), 
    #############################
        html.Div([
            html.Label('Initial Point Position_________: '),   
            dcc.Input(
                id='input-a',
                type='number',
                value=-np.pi,  # Initial value
                debounce=True   
            ), 
        ]),  
    ############################
        html.Div([
            html.Label('Final Point Position__________: '),   
            dcc.Input(
                id='input-b',
                type='number',
                value=np.pi,  # Initial value
                debounce=True   
            ),
        ]),
        html.Div([
            html.Label('Tolerance_____________________: '),   
            dcc.Input(
                id='input-tol',
                type='number',
                value=1e-6,  # Initial value
                debounce=True  # Delay the callback until typing stops
            ), 
        ]), 

        
    ####################################################################################
    html.Br(), 
            dcc.Dropdown(
                id='dropdown-cas',
                options=[
                        {'label': 'Fletcher–Reeves',            'value': 'FLETCHER-REEVES',      'style':  dropdown_options_style},
                        {'label': 'Polak–Ribière',              'value': 'POLAK-RIIERE',         'style':  dropdown_options_style},
                        {'label': 'Hestenes-Stiefel',           'value': 'HESTENES-STIEFEL',     'style':  dropdown_options_style},
                        ],
                value='',
                placeholder='Select the beta formula',
                style=box_style,
                searchable=True,
            ) ,
    html.Br(),
    html.Hr(style={'border-color': 'white'}),

                ############################################################################################################
            html.Div([
            html.Div(id='output-fig-bolza' ), 
        ],
        style={'textAlign': 'center',
                    'color': 'white',
                    'background-color' : 'black',
                    'font-size': 20,
                    'margin':'auto',
                    'width': '50%', 
                    }
        ),
            html.Br(),

            ]),
        ]),        
        
    ],
                style={'textAlign': 'center',
                            'color': 'white',
                            'background-color' : 'black',
                            'font-size': 20
                            }
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








@app.callback([ 
        Output('output-fig-bolza', 'children'),
    ],
    [
        Input('dropdown-cas' , 'value'),
        Input('input-m'   , 'value'),
        Input('input-n'    , 'value'), 
        Input('input-a'    , 'value'),  
        Input('input-b'       , 'value'),
        Input('input-tol' , 'value'),
        # Input('min-max-store', 'data'),
        # Input('my-slider', 'value')
        ],
        prevent_initial_call=True
              )



def update_output(cas,m,n,a,b,tol): 
  
 
        # a=-np.pi
        # b=np.pi

        # n =4
        # m = 3
        x = np.random.rand(n)
        y0 = np.random.rand(m) 


        Inter = {'a':a,
                 'b':b}

        # cas = 'POLAK-RIIERE'
        y0 = np.random.rand(m)
        y = ConjugateGradient(poly_bolza, x, y0, cas, Inter, tol)
 
        xx = np.linspace(Inter['a'],Inter['b'],1000)
        yy = uFun(y,xx)  

        bheight = 600
        bwidth = 650

        fig = dcc.Graph( figure =plot_fun(xx,yy,bheight,bwidth) )  


        return  [fig]



# Run the app
if __name__ == '__main__':
    app.run_server(  debug=False)