"""Helper functions and command line parser for plot.py."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import argparse

sns.set()
sns.set_context('paper')

plot_parser = argparse.ArgumentParser()

plot_parser.add_argument('--import_path',help='import path',
    type=str,default='./logs')
plot_parser.add_argument('--on_file',help='file with on-policy data',type=str)
plot_parser.add_argument('--gpi_file',help='file with GPI data',type=str)
plot_parser.add_argument('--save_path',help='save path',
    type=str,default='./figs')
plot_parser.add_argument('--save_name',
    help='file name to use when saving plot',type=str,default='userplot')
plot_parser.add_argument('--metric',
    help='metric to plot',type=str,default='J_tot')
plot_parser.add_argument('--window',
    help='number of steps for plot smoothing',type=int,default=2048*100)
plot_parser.add_argument('--timesteps',help='number of steps to plot',
    type=float,default=1e6)
plot_parser.add_argument('--interval',help='how often to plot data',
    type=float,default=5e3)
plot_parser.add_argument('--se_val',
    help='standard error multiplier for plot shading',type=float,default=0.5)

def create_plotparser():
    return plot_parser

def aggregate_sim(results,x,window,metric):
    """Computes running averages for all trials."""
    sim = len(results)
    data_all = np.zeros((sim,len(x)))
    for idx in range(sim):
        log = results[idx]['train']
        samples = np.cumsum(log['steps'])
        x_filter = np.argmax(np.expand_dims(samples,1) 
            >= np.expand_dims(x,0),0)

        try:
            data_total = np.squeeze(log[metric])
        except:
            available = ', '.join(list(log.keys()))
            raise ValueError(
                '%s is not a recognized metric. Available metrics include: %s'%(
                    metric,available))

        if window > 1:
            data_totsmooth = np.convolve(np.squeeze(data_total),
                np.ones(window),'full')[:-(window-1)]        
            len_totsmooth = np.convolve(np.ones_like(data_total),
                np.ones(window),'full')[:-(window-1)]     

            data_ave = data_totsmooth / len_totsmooth   
        else:
            data_ave = data_total

        data_all[idx,:] = data_ave[x_filter]
    
    return data_all 

def open_and_aggregate(filepath,filename,x,window,metric):
    """Returns aggregated data from raw filename."""

    if filename is None:
        results = None
    else:
        with open(os.path.join(filepath,filename),'rb') as f:
            data = pickle.load(f)
        
        M = data[0]['param']['runner_kwargs']['M']
        B = data[0]['param']['runner_kwargs']['B']
        n = data[0]['param']['runner_kwargs']['n']
        if M > 1:
            b_size = n
        else:
            b_size = B * n
        
        window_batch = int(window / b_size)
        
        results = aggregate_sim(data,x,window_batch,metric)
    
    return results

def plot_compare(on_data,gpi_data,x,se_val,save_path,save_name):
    """Creates and saves plot."""
    
    fig, ax = plt.subplots()

    on_color = 'C0'
    gpi_color = 'C1'

    if on_data is not None:
        on_mean = np.mean(on_data,axis=0)
        if on_data.shape[0] > 1:
            on_std = np.std(on_data,axis=0,ddof=1)
            on_se = on_std / np.sqrt(on_data.shape[0])
        else:
            on_se = np.zeros_like(on_mean)

        ax.plot(x/1e6,on_mean,color=on_color,label='On-Policy')
        ax.fill_between(x/1e6,
            on_mean-se_val*on_se,on_mean+se_val*on_se,
            alpha=0.2,color=on_color)
    
    if gpi_data is not None:
        gpi_mean = np.mean(gpi_data,axis=0)
        if gpi_data.shape[0] > 1:
            gpi_std = np.std(gpi_data,axis=0,ddof=1)
            gpi_se = gpi_std / np.sqrt(gpi_data.shape[0])
        else:
            gpi_se = np.zeros_like(gpi_mean)

        ax.plot(x/1e6,gpi_mean,color=gpi_color,label='GPI')
        ax.fill_between(x/1e6,
            gpi_mean-se_val*gpi_se,gpi_mean+se_val*gpi_se,
            alpha=0.2,color=gpi_color)

    ax.set_xlabel('Steps (M)')
    ax.legend()

    # Save plot
    save_date = datetime.today().strftime('%m%d%y_%H%M%S')
    save_file = '%s_%s'%(save_name,save_date)
    os.makedirs(save_path,exist_ok=True)
    save_filefull = os.path.join(save_path,save_file)

    filename = save_filefull+'.png'
    fig.savefig(filename,bbox_inches='tight',dpi=300)