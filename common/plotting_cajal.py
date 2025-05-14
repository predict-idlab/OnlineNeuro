import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

from cajal.nrn.cells import MRG
from cajal.nrn.sources import IsotropicPoint
from cajal.nrn.monitors import StateMonitor
from cajal.units import ms

from typing import Sequence
from pathlib import Path
from .plotting import save_figure
    
colors = ['green','red', 'deepskyblue', 'lime', 'orange', 'magenta', 'cyan', 'yellow', 'black']
    
def plot_setup_3d(mrg:MRG, point_sources:Sequence[IsotropicPoint], labels:list=[], 
                  figsize:tuple[int,int]=(12, 8), dpi:int=300,
                  title: str = "Axon and the Stimulation and Blocking electrode position",
                  save_path: str | Path = "", save_svg: bool=True) -> None:
    
    # Extract 3D coordinates of axon nodes and convert from um to mm
    node_x = np.array([n.x3d(0) for n in mrg.node]) / 1000.0
    node_y = np.array([n.y3d(0) for n in mrg.node]) / 1000.0
    node_z = np.array([n.z3d(0) for n in mrg.node]) / 1000.0

    # Set up the 3D plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the axon as a red line in 3D space
    ax.plot(node_x, node_y, node_z, color='red', label='Axon (MRG)', linewidth=2)

    # Plot each electrode as a green marker in 3D space
    for e, point_source in enumerate(point_sources):
        x = point_source.x.value / 1000.0  # Convert from um to mm
        y = point_source.y.value / 1000.0
        z = point_source.z.value / 1000.0

        # Use provided labels or default to "Electrode"
        label = labels[e] if labels and e < len(labels) else "Electrode"
        
        ax.scatter(x, y, z, color=colors[e], s=80, marker='o', label=label)

    # Add axis labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    # Adjust the viewing angle for better visualization
    ax.view_init(elev=20, azim=30)

    # Add a title and legend, and show the plot
    ax.set_title(label=title)
    ax.legend()
    fig.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, save_svg=save_svg)
    
    plt.show()
        
        
def plot_progression_AP_2D(mrg:MRG, v_rec:StateMonitor, vmin:float=-120, vmax:float=70, 
                           figsize:tuple[int,int]=(9, 5), dpi:int=300,
                           title: str = "AP Progression with activation of stimulating and blocking pulses",
                           save_path: str | Path = "", save_svg: bool=True) -> Axes:
    V = v_rec.v  # This gives you the membrane potential of all nodes over time
    t = v_rec.t  # This gives you all time points recorded

    # Convert time to ms for plotting
    T = t / ms

    # Create meshgrids for node numbers and time steps
    nodes = np.arange(mrg.axonnodes)
    T_2D, N_2D = np.meshgrid(T, nodes, indexing='ij')  # shape (#time_steps, #nodes)
    V_t = V.T

    # Create a 2D figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot the 2D surface (using imshow or pcolormesh)
    c = ax.pcolormesh(T_2D, N_2D , V_t, cmap='jet', vmin=vmin, vmax=vmax)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Node Number')
    ax.set_title(label=title)

    # Add colorbar with label
    fig.colorbar(c, ax=ax, shrink=0.5, aspect=10, label='Membrane Potential (mV)')
    ax.grid()
    fig.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, save_svg=save_svg)
        
    return ax
    
    
def plot_progression_AP_3D(mrg, v_rec, vmin=-120, vmax=70, figsize:tuple[int,int]=(10, 6), dpi:int=300,
                           save_path: str | Path = "", save_svg: bool=True) -> None:
    V = v_rec.v
    t = v_rec.t  # This gives you all time points recorded

    # Convert time to ms for plotting
    T = t / ms

    # Create meshgrids for node numbers and time steps
    nodes = np.arange(mrg.axonnodes)
    T_2D, N_2D = np.meshgrid(T, nodes, indexing='ij')  # shape (#time_steps, #nodes)
    V_t = V.T

    # Create a 3D figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(N_2D, T_2D, V_t, cmap='jet', edgecolor='none', alpha=0.8,
                           vmin=vmin, vmax=vmax)

    ax.set_zlim(vmin, vmax)
    ax.set_xlabel('Node Number')
    ax.set_ylabel('Time (ms)')
    #ax.set_zlabel('Membrane Potential (mV)')
    ax.set_title('AP Progression with with activation of stimulation and blocking pulse')
    # Set the view to top-down (90-degree elevation)
    ax.set_zticks([])
    ax.view_init(elev=90, azim=-90)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Membrane Potential (mV)')

    fig.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, save_svg=save_svg)
        
    plt.show()

    
def plot_stim_and_block(stim, block, v_rec, figsize:tuple[int,int]=(9, 5), dpi:int=300,
                        save_path: str | Path = "", save_svg: bool=True) -> None:  

    t = v_rec.t 
    T = t / ms # time axes (X-as)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(T, np.asarray(stim(t=v_rec.t)), label='stimulation')
    ax.plot(T, np.asarray(block(t=v_rec.t)), label='block')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(f'Electrode waveforms during stimulation; \n Block parameters: Pulse Width: {block.pw}, Amplitude: {block.amp}' 
              f' Delay: {block.delay}')
    ax.legend(loc="upper left")
    ax.grid()
    
    fig.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, dpi=dpi, save_svg=save_svg)
        
    plt.show()
    

def plot_waveform(waveform, v_rec, label: str, ax: Axes|None = None,
                  figsize: tuple[int, int] = (9, 5), dpi: int = 300,
                  title: str = "Stimulation waveform(s)",
                  save_path: str | Path = "", save_svg: bool = True) -> Axes:
    
    t = v_rec.t
    T = t / ms  # time axis in ms
    y = np.asarray(waveform(t=v_rec.t))  # waveform values
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.plot(T, y, label=label)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(label=title)
    ax.legend(loc="upper right")
    ax.grid()

    if save_path:
        fig_to_save = ax.figure
        save_figure(fig_to_save, save_path, dpi=dpi, save_svg=save_svg)
        
    return ax

def plot_first_detection(mrg, v_rec, stim_delay, meas_position, propagation_delay,
                         figsize=(5,3), dpi=90,title="AP propagation") -> Axes:
    ax = plot_progression_AP_2D(mrg, v_rec, figsize=figsize, dpi=dpi,
                                title=title)
    ax.axvline(stim_delay, color='k',label='Pulse init', linestyle='--')
    ax.axhline(meas_position, color='orange',label='Meas pos.', linestyle=':')
    ax.scatter(stim_delay+propagation_delay, meas_position, marker='x', color='r', label='AP detection')

    ax.legend()

    return ax
