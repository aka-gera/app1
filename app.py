import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row([
                dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Nav([
                        dbc.NavLink(page["name"], href=page["path"])
                        for page in dash.page_registry.values()
                        if not page["path"].startswith("/app")
                    ])
            ])
        ],
        fluid=True, 
    ),
    dark=True,
    color='dark'
)

app.layout = dbc.Container([header, dash.page_container], fluid=False)

# if __name__ == '__main__':
# 	app.run_server()
# Run the app
if __name__ == '__main__': 
    app.run_server(   debug=False)#, dev_tools_ui=False, dev_tools_props_check=False)
