""" Raw import of CAm's animation maker script so that I dont loose it. I overrode the file-type to text."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import ICRS, Galactic, GalacticLSR
import astropy.units as u
from astropy.stats import sigma_clip
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from skimage import transform
import yaml
from astropy.io import fits
import math

data, header = fits.getdata("/Users/cam/Downloads/vergely_3D_Dust.fits",
                            header=True)





def open_yaml(filepath):
    with open(filepath, 'r') as stream:
        yaml_data = yaml.safe_load(stream)
    return yaml_data


def generate_init_fig_props():

    layout_file = open_yaml('yaml_files/3d_layout.yaml')
    layout_dict = layout_file['3d_galactic_layout']

    # set plot ranges

    figure = layout_file['initial_figure']
    figure['layout'] = layout_dict


    slider_dict = open_yaml(
        'yaml_files/sliders.yaml')['3d_galactic_slider']
    camera = open_yaml(
        'yaml_files/cameras.yaml')['3d_galactic_camera']
    figure['layout']['scene_camera'] = camera
    #figure['layout']['sliders'] = [slider_dict]

    return (figure, slider_dict)


def vergely_dust():

    data, header = fits.getdata("/Users/cam/Downloads/vergely_3D_Dust.fits",
                                header=True)
    downscale = 6.
    downcube = transform.pyramid_reduce(data,
                                        downscale=downscale,
                                        multichannel=False)
    x_origin, y_origin, z_origin = header['CRVAL1'], header['CRVAL2'], header[
        'CRVAL3']

    x_y_step = math.ceil(601 / downscale)
    z_step = math.ceil(81 / downscale)

    X, Y, Z = np.mgrid[x_origin:3000:x_y_step * 1j,
                       y_origin:3000:x_y_step * 1j, z_origin:400:z_step * 1j]
    X = X.flatten().astype(int)
    Y = Y.flatten().astype(int)
    Z = Z.flatten().astype(int)
    downcube = downcube.T.flatten()


    cut_value = 6000



    x_y_condition = np.where((X >= -cut_value) & (X <= cut_value) & (Y >= -cut_value)
                             & (Y <= cut_value))
    X_cut = X[x_y_condition]
    Y_cut = Y[x_y_condition]
    Z_cut = Z[x_y_condition]
    downcube_cut = downcube[x_y_condition]

    vol_plot = go.Volume(
        x=X_cut,
        y=Y_cut,
        z=Z_cut,
        value=downcube_cut,
        flatshading=True,
        opacity=.7,
        #isomin=np.percentile(downcube, 25),
        #isomax=np.percentile(downcube, 75),
        #isomin=150e-6,
        #isomax=1e-4,
        isomin=500e-6,
        showscale=False,
        colorscale='gray',
        opacityscale='max',
        #opacityscale = [[0, 1], [0.5, 0.2], [1, 1]],
        reversescale=True,
        surface=dict(show=True, count=8),
        spaceframe=dict(show=True),  #,
        contour=dict(show=False, width=8),
        hoverinfo='skip',
        visible = 'legendonly',
        name = 'Vergely+2022 Dust',
        showlegend = True
        )


    return vol_plot


def create_figure(df, key_animate, value_step, outfilename, cumulative = False, units = None):

    values = np.arange(df[key_animate].min(), df[key_animate].max(), value_step)

    figure, slider_dict = generate_init_fig_props()

    for i in range(len(values) - 1):
        frame = {'data': [], 'name': str(values[i])}

        if cumulative:
            dfv = df.loc[df[key_animate] <= values[i]]
        else:
            dfv = df.loc[df[key_animate].between(values[i], values[i + 1])]

        x = dfv['x'].values
        y = dfv['y'].values
        z = dfv['z'].values


        scatter_v = go.Scatter3d(
            x = x,
            y = y,
            z = z,
            mode = 'markers',
            marker = dict(
                color = dfv[key_animate].values,
                colorscale = 'viridis',
                reversescale = True,
                colorbar = dict(
                    title = units,
                    len = .3,
                    orientation = 'v',
                    thickness = 20.,
                    x = 1.
                                ),
                #cmin = 0,
                #cmax = 200,
                size = 3.5,
                opacity = 1.,
                symbol = 'circle'
            ),
            hovertext = dfv['cluster'].values,
            name = 'Clusters with {} <= {}'.format(key_animate, np.round(values[i],2))
        )

        scatter_rest = go.Scatter3d(
            x = df['x'].values,
            y = df['y'].values,
            z = df['z'].values,
            mode = 'markers',
            marker = dict(
                color = 'lightgray',
                size = 1.,
                opacity = 1.,
                symbol = 'circle'
            ),
            hovertext = df['cluster'].values,
            name = 'Clusters'
        )

        scatter_sun = go.Scatter3d(
            x = [0],
            y = [0],
            z = [0],
            mode = 'markers',
            marker = dict(
                color = 'yellow',
                size = 3,
                opacity = 1.,
                symbol = 'circle'
            ),
            hovertext = 'Sun',
            name = 'Sun'
        )
        
        slider_step = {
            'args': [[values[i]], {
                'frame': {
                    'duration': 5,
                    'redraw': True
                },
                'mode': 'immediate',
                'transition': {
                    'duration': 100
                }
            }],
            'label':
            np.round(values[i], 1),
            'name':
            'Time',
            'method':
            'animate'
        }

        if i == 0:
            figure['data'] = [scatter_v, scatter_rest, scatter_sun]

        frame['data'] = [scatter_v, scatter_rest, scatter_sun]
        figure['frames'].append(go.Frame(frame))
        slider_dict['steps'].append(slider_step)


    if cumulative:
        slider_string = '{} ({}) <= '.format(key_animate, units)
    else:
        slider_string = '{} ({}) = '.format(key_animate, units)
    slider_dict['currentvalue']['prefix'] = slider_string


    figure['data'].append(vergely_dust())
    figure['layout']['sliders'] = [slider_dict]
    #figure['layout']['template'] = 'plotly'
    fig = go.Figure(figure)
    fig.update_coloraxes(colorbar = dict(len = 2))
    fig.write_html(outfilepath + outfilename, auto_play = False)







filepath = '/Users/cam/Downloads/' # change this to where you are loading the data in from
outfilepath = '/Users/cam/Downloads/' # change this to where you want your plots to be saved

df_avg = pd.read_csv(filepath + 'maria_catalog_cam_grouped.csv') # a dataframe storing the average properties of each cluster

# Make the cuts on our data frame
df_avg = df_avg.loc[
    (df_avg['age_myr'] <= 100) &     # less than or equal to 60 Myr
    (df_avg['ref'] != 'Liu+2019')  # not from Liu+2019, we don't trust this catalog
    ] 
create_figure(df_avg, 'age_myr', .5, outfilename = 'maria_age_animation.html', cumulative = True, units = "Myr")


df = pd.read_csv('/Users/cam/Downloads/hunt_clusters.csv')
df = df.rename(columns={'name' : 'cluster'})
df = df.loc[df['age_myr'] <= 100]
create_figure(df, 'age_myr', .5, outfilename = 'hunt_age_animation.html', cumulative = True, units = "Myr")
