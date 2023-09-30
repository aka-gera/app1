import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
 
dash.register_page(__name__, title='Publications', name='Publications', order=3)

def layout():
    return html.Div([ 
 html.Br(), 
    dcc.Markdown('# PUBLICATIONS', style={'textAlign': 'center'}), 
 html.Br(), 
 html.Hr(), 
        dbc.Row([
        dbc.Col([
            dcc.Markdown('#### Strong Convergence of Integrators for Nonequilibrium Langevin Dynamics'),
             html.Hr(), 
             dcc.Markdown('with Matthew Dobson'),
             html.Hr(),
            dcc.Markdown('in Molecular Simulation, Volume 45, Issue 11 (2019) 912--920..'),
             html.Hr(), 
            dcc.Markdown("[DOI: 10.1080/08927022.2019.1610950](https://www.tandfonline.com/doi/full/10.1080/08927022.2019.1610950)"),
        ], width=5),
        dbc.Col([
            dcc.Markdown('Abstract'),
            html.Hr(),
            dcc.Markdown('Several numerical schemes are proposed for the solution of Nonequilibrium'
                         ' Langevin Dynamics (NELD), and the strong rate of convergence for each scheme' 
                          ' is analyzed. The schemes considered here employ specialised periodic boundary'
                           ' conditions that deform with the flow, namely Lees-Edwards and Kraynik-Reinelt '
                           'boundary conditions and their generalisations. We show that care must be taken '
                           'when implementing standard stochastic integration schemes with these boundary '
                           'conditions in order to avoid a breakdown in the strong order of convergence.',
                         className='ms-3'), 
        ], width=5)
    ], justify='center'),
    html.Hr(),
   ################################################################################################################# 
        dbc.Row([
        dbc.Col([
            dcc.Markdown('#### Simple Periodic Boundary Conditions for Molecular Simulation of Uniaxial Flow'),
             html.Hr(), 
             dcc.Markdown('with Matthew Dobson'),
             html.Hr(),
            dcc.Markdown('in Journal of Computational Physics, Volume 473, Issues (2023), pages 111740'),
             html.Hr(), 
            dcc.Markdown("[DOI:  10.1016/j.jcp.2022.111740](https://www.sciencedirect.com/science/article/pii/S0021999122008038)"),
        ], width=5),
        dbc.Col([
            dcc.Markdown('Abstract'),
            html.Hr(),
            dcc.Markdown('We present rotating periodic boundary conditions (PBCs) for the simulation '
                         'of nonequilibrium molecular dynamics (NEMD) under uniaxial stretching flow (USF) '
                         'or biaxial stretching flow (BSF). Such nonequilibrium flows need specialized PBCs '
                         'since the simulation box deforms with the background flow. The technique builds on '
                         'previous models using one or two lattice remappings, and is simpler than the PBCs '
                         'developed for the general three dimensional flow. For general three dimensional flows, '
                         'Dobson [1](https://www.sciencedirect.com/science/article/abs/pii/S0021999122008038?via%3Dihub#br0010) '
                         'and Hunt [2](https://www.tandfonline.com/doi/full/10.1080/08927022.2015.1051043) proposed schemes '
                         'which are not time-periodic since they use '
                         'more than one automorphism remapping. This paper presents a single automorphism remapping '
                         'PBCs for USF and BSF which is time periodic up to a rotation matrix and has better minimum '
                         'lattice spacing properties.',
                         className='ms-3'), 
        ], width=5)
    ], justify='center'),
    html.Hr(),
       ################################################################################################################# 
        dbc.Row([
        dbc.Col([
            dcc.Markdown('#### Convergence of Nonequilibrium Langevin Dynamics for Planar Flows'),
             html.Hr(), 
             dcc.Markdown('with Matthew Dobson'),
             html.Hr(),
            dcc.Markdown('in Journal of Statistical Physics volume 190, Article number: 91 (2023)'),
             html.Hr(), 
            dcc.Markdown("[DOI:  10.1007/s10955-023-03109-3](https://link.springer.com/article/10.1007/s10955-023-03109-3)"),
        ], width=5),
        dbc.Col([
            dcc.Markdown('Abstract'),
            html.Hr(),
            dcc.Markdown('We prove that incompressible two-dimensional nonequilibrium Langevin dynamics (NELD) '
                         'converges exponentially fast to a steady-state limit cycle. We use automorphism '
                         'remapping periodic boundary conditions (PBCs) such as Lees–Edwards PBCs and Kraynik–Reinelt '
                         'PBCs to treat respectively shear flow and planar elongational flow. The convergence is shown '
                         'using a technique similar to (Joubaud et al. in J Stat Phys 158:1–36, 2015).',
                         className='ms-3'), 
        ], width=5)
    ], justify='center'),
    html.Hr(),
 
 html.Br(), 
    dcc.Markdown('# Manuscripts', style={'textAlign': 'center'}), 
 html.Br(), 
 html.Hr(),       
   ################################################################################################################# 
        dbc.Row([
        dbc.Col([
            dcc.Markdown('#### Math Systems for Diagnosis and Treatment of Breast Cancer'),
             html.Hr(), 
             dcc.Markdown('with C. Amorin, G. P. Andrade, S. Castro-Pearson,  B. Iles, D. Katsaros, T. Mullen, S. Nguyen, O. Spiro, and M. Sych'),
             html.Hr(), 
            dcc.Markdown("[UMass Amherst Department of Mathematics & Statistics Newsletter](https://www.math.umass.edu/sites/www.math.umass.edu/files/newsletters/2017mathnewsletter.pdf) (2017)"),
        ], width=5),
        dbc.Col([
            dcc.Markdown('Motivation'),
            html.Hr(),
            dcc.Markdown('According to statistics from the CDC, breast cancer is,'
                         'the most common cancer diagnosis among women in the'
                         'United States. Currently, radiology is an expert-based field'
                         'where automated tools are at best relegated to the role of'
                         '“second reader.” Early detection is an enormously important'
                         'part of breast cancer treatment, so our goal in this project'
                         'is to create a machine learning pipeline for detection and'
                         'diagnosis from mammogram images. We also model the'
                         'growth and treatment of tumors using a system of ODEs.'
                         'Due to a lack of human data, this last part of the pipeline'
                         'uses data from experiments on lab mice, and is not restricted'
                         'to breast cancer',
                         className='ms-3'), 
        ], width=5)
    ], justify='center'),
       ################################################################################################################# 
        dbc.Row([
        dbc.Col([
            dcc.Markdown('#### Multiple Scale Modeling For Predictive Material Deformation Analysis'),
             html.Hr(), 
             dcc.Markdown('with  R. Aronow, A. d. Silva, R. Dennis, D. Katsaros, M. Sych, and R. Touret'),
             html.Hr(), 
            dcc.Markdown("[UMass Amherst Department of Mathematics & Statistics Newsletter](https://www.umass.edu/mathematics-statistics/sites/default/files/newsletters/2018_umass_math_newsletter_210.pdf) (2018)"),
        ], width=5),
        dbc.Col([
            dcc.Markdown('Motivation'),
            html.Hr(),
            dcc.Markdown(' Material deformation and stress-strain is an'
                         'active area of mathematical modeling relevant to industrial'
                         'and research-oriented materials science. It is important to'
                         'take variations in material properties into account in these'
                         'models. Multi-scale models that incorporate inhomogeneity'
                         'were studied and modeling frameworks that address this'
                         'need were created and tested. Incorporating variations in'
                         'material properties at the micro scale resulted in significantly'
                         'different predictions of material deformation under similar'
                         'loading. Variations in material properties were accounted'
                         'for through averaging over stresses in representative volume'
                         'elements (RVEs).',
                         className='ms-3'), 
        ], width=5)
    ], justify='center'),
])
