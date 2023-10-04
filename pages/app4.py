import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from .side_bar import sidebar

dash.register_page(__name__, title='NELD Limte Cycle', name='NELD Limit Cycle',order=2)
 




 






import numpy as np
from scipy.linalg import expm
import time 
import plotly.graph_objects as go
import scipy
from scipy import stats
 


tcouleur = 'plotly_dark'
bcouleur = 'navy'
fcouleur = 'white'
fsize = 20
   

np.random.seed(0)


def Parameter(flow, epsilon, nPart, rcut, N, Nperiod):

    if nPart < 2:
        raise ValueError("The number of particles is invalid")


    if flow == 'eld':
        if nPart <= 2:
            a = 10
        else:
            a = 2*nPart
        A = np.array([[0,0,0],[0,0,0],[0,0,0]])
        invL0 = np.eye(3)/a
        Y = A
        Yoff = np.zeros((3,3))
        Sigma = 1
        dim = 2
    elif flow == 'shear':
        # Shear flow case with LE
        if nPart <= 2:
            a = 10
        else:
            a = 2*nPart
        A = epsilon * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        invL0 = np.eye(3) / a
        Y = A
        Yoff = np.zeros((3, 3))
        Sigma = epsilon
        dim = 2
    elif flow == 'pef':
        # PEF case with KR
        if nPart <= 4:
            a = 20
        else:
            a = 6*nPart

        A = epsilon * np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        M = np.array([[2, -1, 0], [-1, 1, 0], [0, 0, 1]])
        _,V = np.linalg.eig(M)
        V = np.dot([[-1, 0, 0],[0, -1, 0],[0, 0, 1]], V[:, [1, 0, 2]])
        Y = np.log(np.diag(np.dot(np.dot(np.linalg.inv(V), M), V)))
        Yoff = np.zeros((3, 3))
        invL0 = V / np.abs(np.linalg.det(V)) ** (1/2) / a 
        Sigma = -epsilon/Y[0]
 
    pbc = {}

    pbc['flow'] = flow
    pbc['L0inv'] = invL0
    pbc['L0'] = np.linalg.inv(invL0)
    pbc['Linv'] = pbc['L0inv']
    pbc['L'] = pbc['L0']
    pbc['A'] = A
    pbc['Y'] = Y
    pbc['Yoff'] = Yoff
    pbc['Sigma'] = Sigma
    pbc['T'] = 1/abs(Sigma)
    pbc['theta'] = 0
    pbc['theta1'] = 0
    pbc['n'] = 0
    pbc['dt'] = pbc['T'] / N
    pbc['N'] = N
    pbc['Nperiod'] = Nperiod

    part = {}


    part['q'] = np.zeros((3,nPart))
    part['qDist'] = np.zeros((3,nPart))
    part['p'] = np.zeros((3,nPart))
    part['p'][2,:nPart] = np.zeros(nPart)
    part['f'] = np.zeros((3,nPart))
    part['ff'] = np.zeros(1)
    part['G'] = np.zeros((3,nPart))

    sav = {}

    sav['Q1'] = np.zeros((N,Nperiod))
    sav['Q2'] = np.zeros((N,Nperiod))
    sav['F'] = np.zeros((N,Nperiod))


    dim = A.shape[0]-1

    param = {}

    param['sigm'] = 4
    param['eps'] = 1
    param['rcut'] = rcut
    param['dim'] = dim
    param['gamma'] = 0.1
    param['beta'] = 1
    param['a'] = a
    param['nPart'] = nPart
    param['Mmax'] = int(np.ceil(a/rcut*nPart))
    param['vol'] = a**3 * nPart

    Clist_Mmax = int(np.floor(a*dim/param['rcut']))
    Clist_head = np.zeros((param['Mmax']**3,1))
    Clist_list = np.zeros((nPart,1))
    Clist_mc = np.zeros((1,3))
    Clist_da = np.zeros((1,3))
    Clist_nL = np.zeros((1,3))
    Clist_c = np.zeros((1))
    Clist_lc = np.zeros((1,3))
    Clist_region = np.zeros((1,3))
    Clist_M = np.zeros((1,3))

    Clist = {'Mmax': Clist_Mmax,
            'head': Clist_head,
            'lis': Clist_list,
            'mc': Clist_mc,
            'da': Clist_da,
            'nL': Clist_nL,
            'c': Clist_c,
            'lc': Clist_lc,
            'region': Clist_region,
            'M': Clist_M}
    return pbc, param, Clist, part, sav

 
def paramFig(PBC, sbox):
    mm = 100
    lSpace = np.linspace(1, np.exp(1), mm)
    II = np.ones(mm)
    I0 = np.zeros(mm)
    III = II - np.log(lSpace)

    dat = {
        'mapp': [II, III, III],
        'MainBoxColor': [1, 0, 0],
        'MainBoxEdge': ':',
        'MainBoxOpaque': 0.05,
        'MainBoxMarkerWidth': 1,
        'Color': 'r',
        'GridEdge': 'o',
        'GridColor': [0, 0, 0],
        'GridMarkerWidth': 2,
        'ft': 20,
        'AxisWidth': 3,
        'AxisColor': 'b',
        'aa': 15,
        'bb': 1
    }

    if PBC == 'eld':
        dat['mapp'] = [II, I0, I0]
        dat['Angle'] = [0, 90]
        dat['posTextX'] = -2
        dat['posTextY'] = 4
        dat['posTextZ'] = 7
        dat['aa'] = 0
        dat['bb'] = 20
        dat['xmin'] = -1.1 * sbox
        dat['xmax'] = 2.1 * sbox
        dat['ymin'] = -1.01 * sbox
        dat['ymax'] = 2.01 * sbox
        dat['zmin'] = -1.9 * sbox
        dat['zmax'] = 1.17 * sbox
        dat['center'] = [0, 0, 1]
        dat['radius'] = 1.5
        dat['centerOff'] = 0
    elif PBC == 'shear':
        dat['mapp'] = [II, I0, I0]
        dat['Angle'] = [0, 90]
        dat['posTextX'] = -2
        dat['posTextY'] = 4
        dat['posTextZ'] = 7
        dat['aa'] = 0
        dat['bb'] = 20
        dat['xmin'] = -1.1 * sbox
        dat['xmax'] = 2.1 * sbox
        dat['ymin'] = -1.01 * sbox
        dat['ymax'] = 2.01 * sbox
        dat['zmin'] = -1.9 * sbox
        dat['zmax'] = 1.17 * sbox
        dat['center'] = [0, 0, 1]
        dat['radius'] = 1.5
        dat['centerOff'] = 0
    elif PBC == 'pef':
        dat['mapp'] = [II, I0, I0]
        dat['Angle'] = [180, 90]
        dat['posTextX'] = 0
        dat['posTextY'] = -5
        dat['posTextZ'] = 7
        dat['aa'] = 0
        dat['bb'] = 20
        dat['xmin'] = -2.7 * sbox
        dat['xmax'] = 2.3 * sbox
        dat['ymin'] = -1.7 * sbox
        dat['ymax'] = 1.7 * sbox
        dat['zmin'] = -1.9 * sbox
        dat['zmax'] = 1.17 * sbox
        dat['center'] = [-0.75, -0.3, 1]
        dat['radius'] = -1.75
        dat['centerOff'] = 0.4

    dat['Axi'] = [dat['xmin'], dat['xmax'], dat['ymin'], dat['ymax'], dat['zmin'], dat['zmax']]

    xlength = 2.5
    ylength = 2  # Assuming there was an error in the MATLAB code where "2.5" shouldn't be here

    dat['xmax'] = sbox# xlength * sbox  # size of x dimension of graph
    dat['ymax'] = sbox# ylength * sbox  # size of y dimension of graph
    dat['zmax'] = 1  # size of z dimension of graph

    aa = dat['xmax']+xlength * sbox
    bb = dat['ymax']+ylength * sbox
    cc = dat['zmax']

    x = np.arange(-aa, aa + 1)
    y = np.arange(-bb, bb + 1)
    z = np.arange(-cc, cc + 1)

    xx, yy, zz = np.meshgrid(x, y, z)

    dat['PP'] = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))  
    

    return dat
 

 
def data_replicas(L, q, dat, param):
    # Input
    # L : simulation box in 2 dimensions
    # q : position of the particles
    # dat : unit Lattice grid

    # Output
    # qq : position of the particles in the simulation box and its replicas
    # Lk : simulation box in 3 dimensions
    # LB : simulation box with its replicas

    LL = np.dot(L, dat['PP'].T)
    # inds =  (LL[0, :] < dat['xmax']) & (LL[0, :] > -dat['xmax']) \
    #       & (LL[1, :] < dat['ymax']) & (LL[1, :] > -dat['ymax']) \
    #       & (LL[2, :] < dat['zmax']) & (LL[2, :] > -dat['zmax'])

    mm = q.shape[1]
    LB = LL#[:, inds] 
    nn = LB.shape[1] 
    qL = LB[0:3, :] 

    qTemp = qL + np.tile(q[:, 0], (1, nn)).reshape(nn,3).T 
    indsTemp = (qTemp[0, :] < dat['xmax']) & (qTemp[0, :] > -dat['xmax']) \
             & (qTemp[1, :] < dat['ymax']) & (qTemp[1, :] > -dat['ymax'])  \
             & (qTemp[2, :] < dat['zmax']) & (qTemp[2, :] > -dat['zmax']) 
    
    qq =qTemp[:,indsTemp]

    for i in range(mm-1): 
        qTemp = qL + np.tile(q[:, i+1], (1, nn)).reshape(nn,3).T 
        indsTemp = (qTemp[0, :] < dat['xmax']) & (qTemp[0, :] > -dat['xmax']) \
            & (qTemp[1, :] < dat['ymax']) & (qTemp[1, :] > -dat['ymax'])  \
            & (qTemp[2, :] < dat['zmax']) & (qTemp[2, :] > -dat['zmax']) 
        qq =np.hstack((qq, qTemp[:,indsTemp]))
        
    return qq



def MyRound(x):
  return x - np.round(x)

def initializez(X, param, pbc):
    if param['dim'] == 3:
        ll = 0
        for l in range(param['nPart']):
            i = l+1
            j = i+1
            X.q[:,ll] = [(0.5 + i-0.5*param['nPart'])/param['nPart'],
                         (0.5 + j-0.5*param['nPart'])/param['nPart'],
                         (0.5 + l-0.5*param['nPart'])/param['nPart']]
            ll += 1
    else:
        ll = 0
        for l in range(param['nPart']):
            j = l+1
            X['q'][:,ll] = [(0.5 + l-0.5*param['nPart'])/param['nPart'],
                         (0.5 + j-0.5*param['nPart'])/param['nPart'],
                         0]
            ll += 1

    X['q'] = np.dot(pbc['L'], X['q'])
    X['q'][0:param['dim'],:] = X['q'][0:param['dim'],:] + 0.05 * np.random.randn(param['dim'], param['nPart'])
    X['p'] = np.dot(pbc['A'], X['q'])
    X['p'][0:param['dim'],:] = X['p'][0:param['dim'],:] + np.sqrt(1/param['beta']) * np.random.randn(param['dim'], param['nPart'])
    return X

def EmEulerian(X, pbc, param, Z):
    # Update position
    X['q'] = X['q'] + (X['p'] + np.dot(pbc['A'] , X['q'])) * pbc['dt']

    # Compute force
    X = ComputeForceEulerian(X, param, pbc)
    # X = ComputeForceEulerianCell(X, param, pbc, Z)
    # Update momentum
    X['G'][:param['dim'], :param['nPart']] = np.sqrt(2 * pbc['dt'] * param['gamma'] / param['beta']) * np.random.randn(param['dim'], param['nPart'])

    X['p'] = X['p'] + X['f'] * pbc['dt'] - param['gamma'] * X['p'] * pbc['dt'] + X['G']

    # Remap position
    X['q'], pbc = Remap_Eulerian_q(X['q'], pbc)

    return X, pbc

def Remap_Eulerian_q( q, pbc):

    pbc['L'] = np.dot(  MyExp(pbc['Y']*pbc['theta']) , pbc['L0'])
    pbc['Linv'] = np.dot( pbc['L0inv'], MyExp(-pbc['Y']*pbc['theta']))
    q = np.dot(pbc['L'], MyRound(np.dot(pbc['Linv'], q)))
    return q, pbc


def MyExp(M):
    if len(M.shape) > 1:
        f = expm(M)
    else:
        f = np.diag(np.exp(M))
    return f

def fLJ(rr, param):
  if rr > param['rcut']:
      p = 0.0
  else:
      p = 4 * param['eps'] * ((12 * param['sigm'] ** 6) / rr ** 7 - (12 * param['sigm'] ** 12) / rr ** 13)
  return p

def ComputeForceEulerian(X, param, pbc):
    mm1 = 1
    mm2 = 1
    X['f'][:param['dim'],:] = np.zeros((param['dim'], param['nPart']))
    for i in range(param['nPart']-1):
        for j in range(i+1, param['nPart']):
            X['qDist'] = X['q'][:, i] - X['q'][:, j]
            X['qDist'], _ = Remap_Eulerian_q(X['qDist'], pbc)
            normqD = np.linalg.norm(X['qDist'])
            ff = fLJ(normqD, param)
            X['f'][:, i] = X['f'][:, i] - ff * X['qDist'] / normqD
            X['f'][:, j] = X['f'][:, j] + ff * X['qDist'] / normqD
            if mm1 < abs(ff):
                mm1 = abs(ff)
                mm2 = ff

    X['ff'] = mm2
    return X

def Simulation(X, pbc, param, lis, sav, animation):


  X = initializez(X, param, pbc)
  X,pbc = EmEulerian(X,pbc,param,lis)
  tic = time.time()
  # Run the simulation
  igif = 1
  for j in range(pbc['Nperiod']):
      fmax = 1e-16
      for i in range(pbc['N']):
          # print(i,X['ff']  )
          sav['Q1'][i, j] = X['qDist'][0]
          sav['Q2'][i, j] = X['qDist'][1]
          sav['F'][i, j] = X['ff']
          t = 1e-3 * round(1e3 * (i - 1) * pbc['dt'])
          X, pbc = EmEulerian(X, pbc, param, lis)
          pbc['theta1'] = pbc['theta'] + pbc['Sigma'] * pbc['dt']
          pbc['theta'] = pbc['theta1'] - np.floor(pbc['theta1'])
          pbc['n'] = pbc['n'] + pbc['theta'] - pbc['theta1']
          if np.abs(fmax) < np.abs(X['ff']):
              fmax = X['ff']
      if np.mod(j, np.round(pbc['Nperiod'] / 10)) == 0:
          time_ = time.time()
          print(f"Period {j} executed in {np.round(1000*(time_-tic)/60)/1000} min")
          print(fmax)
  print('force : ',fmax)
  return sav
 
def HistoryMatrix(sav,Ndata,ShowTime):

    QQ1 = sav['Q1'].T
    QQ2 = sav['Q2'].T
    xa = np.min(np.min(QQ1))
    xb =  np.max([np.max(np.max(QQ1)),1])
    ya = np.min(np.min(QQ2))
    yb =  np.max([np.max(np.max(QQ2)),1])
    xedges = np.linspace(xa, xb, Ndata+1)
    yedges = np.linspace(ya, yb, Ndata+1)

    minColorLimit = 0
    maxColorLimit = 0
    histMa = np.zeros((Ndata, Ndata, len(ShowTime)))

    for k in ShowTime:
        # histmat, _ , _,_ = binned_statistic_2d(QQ1[:, k], QQ2[:, k], values=None, statistic='count', bins=[xedges, yedges])
        ret = stats.binned_statistic_2d(QQ1[:, k], QQ2[:, k], values=None, statistic='count', bins=[xedges, yedges])
        histmat = ret.statistic
        minColorLimit = min([np.min(np.min(histmat)), minColorLimit])
        maxColorLimit = max([np.max(np.max(histmat)), maxColorLimit])
        histMa[:, :, k] = histmat.T

    return histMa,minColorLimit,maxColorLimit,xa,xb,ya,yb
  
def plot_history_matrixxy(sav,dt,bheight,bwidth):
  
  QQ1 = sav['Q1'].T
  QQ2 = sav['Q2'].T
  xa = np.min(np.min(QQ1))
  xb = np.max([np.max(np.max(QQ1)),1])
  ya = np.min(np.min(QQ2))
  yb = np.max([np.max(np.max(QQ2)),1]) 

  minColorLimit = 0
  maxColorLimit = 1000
  
  m = sav['Q1'].shape[0]
  frames = []
  frame_titles = []


  x = sav['Q1'][0, :]
  y = sav['Q2'][0, :]
 
  trace = go.Histogram2d(
      x=x,
      y=y,
      autobinx=False,
      xbins=dict(start=xa, end=xb, size=0.1),
      autobiny=False,
      ybins=dict(start=ya, end=yb, size=0.1),
      colorscale='Viridis',
      showscale=True,
      colorbar=dict(
          titleside="top",
          tickmode="array",
          tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
  ) 

  fig = go.Figure(data=[trace],
                  layout=dict(xaxis=dict(range=[xa, xb]),
                              yaxis=dict(range=[ya, yb]),
                              showlegend=False, )
                  ) 

  fig.update_layout(
      title=f'Time {0:.2f}'
  )


  for i in range(m):
      x = sav['Q1'][i, :]
      y = sav['Q2'][i, :]
    
      trace = go.Histogram2d(
          x=x,
          y=y,
          autobinx=False,
          xbins=dict(start=xa, end=xb, size=0.1),
          autobiny=False,
          ybins=dict(start=ya, end=yb, size=0.1),
          colorscale='Viridis',
          showscale=True,
          colorbar=dict(
              titleside="top",
              tickmode="array",
              tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
      )

  
      frames.append({'data': [trace]})

      frame_titles.append(f'Time {dt*i:.2f}') 
      # fig.add_trace(trace)



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
      hovermode='closest', 
  )

  for i, frame_title in enumerate(frame_titles):
      frames[i].update(layout=dict(title=f'Time {dt*i:.2f}'))


  fig.update(frames=frames)
  
  fig.update_layout(
      updatemenus=[
          dict(
              type='buttons',
              showactive=False,
              buttons=[
                  dict(label='Play',
                      method='animate',
                      args=[None, dict(frame=dict(duration=0, redraw=True), fromcurrent=True, mode='immediate')]),
                  dict(label='Pause',  # Add a pause button
                      method='animate',
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
              ],
              x=0.1,
              xanchor='right',
              y=1.2,
              yanchor='top',
          )
      ],
  )

  fig.update_xaxes(
      title_text='x1-x2',
      title_font = {"size": 18},
      title_standoff = 25,
      side='bottom')
  fig.update_yaxes(
          title_text = 'y1-y2',
          title_font = {"size": 18},
          title_standoff = 25)
  ###################################################################
  fig.data[0].visible = True

  steps = []
  for i in range(len(fig.data)):
      step = dict(
          method="restyle",
          args=["visible", [False] * len(fig.data)],
          label=str(i),
      )
      step["args"][1][i] = True  # Toggle i'th trace to "visible"
      steps.append(step)

  sliders = [dict(
      active=10,
      currentvalue={"prefix": "Frequency: "},
      pad={"t": 50},
      steps=steps
  )]

 


  fig.update_layout(sliders=sliders)
  #########################################################################


  return fig


 
def plot_history_matrixxy2(sav,dat,param,pbc,bheight,bwidth):
  
  QQ1 = sav['Q1'].T
  QQ2 = sav['Q2'].T
 
# Calculate L matrix
  L =  pbc['L0']

  q1 = QQ1[:, 0].reshape(-1,1)
  q2 = QQ2[:, 0].reshape(-1,1)
  q3 = np.zeros((QQ1.shape[0],1))
   
  qqX = np.hstack((q1, q2, q3)).T
  qq = data_replicas(L, qqX, dat, param) 
  x =qq[0,:].reshape(-1) 
  y =qq[1,:].reshape(-1) 
  xa = -dat['xmax']# np.min(np.min(x))
  xb =  dat['xmax']# np.max(np.max(x)) 
  ya = - dat['ymax']#np.min(np.min(y))
  yb = dat['ymax']#np.max(np.max(y))  

  minColorLimit = 0
  maxColorLimit = 1000

  m = sav['Q1'].shape[0]
  frames = []
  frame_titles = []
 
  # Create a heatmap trace for the frame
  trace = go.Histogram2d(
      x=x,
      y=y,
      autobinx=False,
      xbins=dict(start=xa, end=xb, size=0.1),
      autobiny=False,
      ybins=dict(start=ya, end=yb, size=0.1),
      colorscale='Viridis',
      showscale=True,
      colorbar=dict(
          titleside="top",
          tickmode="array",
          tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
  ) 

  fig = go.Figure(data=[trace],
                  layout=dict(xaxis=dict(range=[xa, xb]),
                              yaxis=dict(range=[ya, yb]),
                              showlegend=False, )
                  ) 

  fig.update_layout(
      title=f'Time {0:.2f}'
  )


  for i in range(m):
      
      theta = i * pbc['dt'] - np.floor(i * pbc['dt'] / pbc['T']) * pbc['T']

    # Calculate L matrix
      L = scipy.linalg.expm(theta * pbc['A']).dot(pbc['L0'])

      q1 = QQ1[:, i].reshape(-1,1)
      q2 = QQ2[:, i].reshape(-1,1) 

      qqX = np.hstack((q1, q2, q3)).T
      qq = data_replicas(L, qqX, dat, param) 
      x =qq[0,:].reshape(-1) 
      y =qq[1,:].reshape(-1)  
    
      trace = go.Histogram2d(
          x=x,
          y=y,
          autobinx=False,
          xbins=dict(start=xa, end=xb, size=0.1),
          autobiny=False,
          ybins=dict(start=ya, end=yb, size=0.1),
          colorscale='Viridis',
          showscale=True,
          colorbar=dict(
              titleside="top",
              tickmode="array",
              tickvals=list(range(int(minColorLimit), int(maxColorLimit)))),
      )

  
      frames.append({'data': [trace]})
      dt =pbc['dt']
      frame_titles.append(f'Time { dt*i:.2f}') 
      # fig.add_trace(trace)



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
      hovermode='closest', 
  )

  for i, _ in enumerate(frame_titles):
      frames[i].update(layout=dict(title=f'Time {dt*i:.2f}'))


  fig.update(frames=frames)
  
  fig.update_layout(
      updatemenus=[
          dict(
              type='buttons',
              showactive=False,
              buttons=[
                  dict(label='Play',
                      method='animate',
                      args=[None, dict(frame=dict(duration=0, redraw=True), fromcurrent=True, mode='immediate')]),
                  dict(label='Pause',  # Add a pause button
                      method='animate',
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
              ],
              x=0.1,
              xanchor='right',
              y=1.2,
              yanchor='top',
          )
      ],
  )

  fig.update_xaxes(
      title_text='x1-x2',
      title_font = {"size": 18},
      title_standoff = 25,
      side='bottom')
  fig.update_yaxes(
          title_text = 'y1-y2',
          title_font = {"size": 18},
          title_standoff = 25)
  ###################################################################
  fig.data[0].visible = True

  steps = []
  for i in range(len(fig.data)):
      step = dict(
          method="restyle",
          args=["visible", [False] * len(fig.data)],
          label=str(i),
      )
      step["args"][1][i] = True  # Toggle i'th trace to "visible"
      steps.append(step)

  sliders = [dict(
      active=10,
      currentvalue={"prefix": "Frequency: "},
      pad={"t": 50},
      steps=steps
  )]

 


  fig.update_layout(sliders=sliders)
  #########################################################################


  return fig




########################## APP BEGIN ###########################################################

shw = 0

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
# Create a dash application Cyborg







def layout():
    return html.Div(
    style={
        'color' : 'black',
        'backgroundColor': 'black',  # Set the background color of the app here
        'height': '100vh'  # Set the height of the app to fill the viewport
    },
    children=[
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1('Nonequilibrium Langevin Dynamics Limit Cycle',
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
            html.Label('Particule Numbers____: '),   
            dcc.Input(
                id='input-neld-nPart',
                type='number',
                value=2,  # Initial value
                debounce=True   
            ),
        ]),
        html.Div([
            html.Label('Box Deformation Rate: '),   
            dcc.Input(
                id='input-neld-epsilon',
                type='number',
                value=1,  # Initial value
                debounce=True  # Delay the callback until typing stops
            ),
        ]), 
    #############################
        html.Div([
            html.Label('Raduis Cut_____________: '),   
            dcc.Input(
                id='input-neld-rcut',
                type='number',
                value=30,  # Initial value
                debounce=True   
            ), 
        ]),  
    ############################
        html.Div([
            html.Label('Total Iterations Time__ : '),   
            dcc.Input(
                id='input-neld-N',
                type='number',
                value=50,  # Initial value
                debounce=True   
            ),
        ]),
        html.Div([
            html.Label('Total Period____________: '),   
            dcc.Input(
                id='input-neld-Nperiod',
                type='number',
                value=100,  # Initial value
                debounce=True  # Delay the callback until typing stops
            ), 
        ]), 

        
    ####################################################################################
    html.Br(), 
            dcc.Dropdown(
                id='dropdown-neld-flow',
                options=[
                        {'label': 'Zero Flow',                          'value': 'eld',     'style':  dropdown_options_style},
                        {'label': 'Shear Flow',                         'value': 'shear',   'style':  dropdown_options_style},
                        {'label': 'Planar Elongational Flow',           'value': 'pef',     'style':  dropdown_options_style},
                        ],
                value='',
                placeholder='Select the type of flow',
                style=box_style,
                searchable=True,
            ) ,

    ],
                style={'textAlign': 'center',
                            'color': 'white',
                            'background-color' : 'black',
                            'font-size': 20
                            }
                ),
html.Br(),
 html.Hr(style={'border-color': 'white'}),

            ############################################################################################################
 
                
    html.Div([
        # html.Div(id='output-neld-fig', style={'display': 'inline-block'}),
        html.Div(id='output-neld-fig2', style={'display': 'inline-block'}),
    ],
    style={'textAlign': 'center',
                'color': 'white',
                'background-color' : 'black',
                'font-size': 20,
                'margin': 'auto',
                'width': '70%', 
                }
    ),

        ]),
     ]),
        html.Br(),
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
        # Output('my-slider', 'min'),
        # Output('my-slider', 'max'),
        # Output('output-neld-fig', 'children'),
        Output('output-neld-fig2', 'children'),
    ],
    [
        Input('dropdown-neld-flow' , 'value'),
        Input('input-neld-nPart'   , 'value'),
        Input('input-neld-epsilon'    , 'value'), 
        Input('input-neld-rcut'    , 'value'),  
        Input('input-neld-N'       , 'value'),
        Input('input-neld-Nperiod' , 'value'),
        # Input('min-max-store', 'data'),
        # Input('my-slider', 'value')
        ],
        prevent_initial_call=True
              )



def update_output(flow,nPart,epsilon,rcut,N,Nperiod): 

        # flow = 'eld'                     # choose the type of the flow (i.e 'eld', 'shear',  or 'pef')
        # nPart = 2                         # Number of particles
        # epsilon = 1.0                     # rate of the deformation of the background flow
        # rcut = 30                         # radius cut
        # N = 30                          # number of steps in a period
        # Nperiod = 100                   # number of periods


        animation = 11                    # 1 to activate the animation simulation box

        pbc, param, lis, X, sav = Parameter(flow, epsilon, nPart, rcut, N, Nperiod)  # get the parameters

        sav = Simulation(X, pbc, param, lis, sav, animation)
        # datF = paramFig(pbc['flow'],param['a'])

        bheight = 600
        bwidth = 600

        fig2 = dcc.Graph( figure = plot_history_matrixxy(sav,pbc['dt'],bheight,bwidth)) 
        # fig = dcc.Graph(figure= plot_history_matrixxy2(sav,datF,param,pbc,bheight,bwidth))


        return  [fig2]








