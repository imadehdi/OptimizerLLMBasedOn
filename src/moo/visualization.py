import numpy as np
from pymoo.visualization.pcp import PCP

def plot_pcp(F: np.ndarray, labels=None, bounds=None, highlight=None, title="Pareto PCP"):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be a 2D array (n_points x n_obj).")
    
    plot = PCP(title=title, labels=labels)
    if bounds is not None:
        plot.normalize_each_axis = False
        plot.bounds = bounds
        
    plot.set_axis_style(color="grey", alpha=0.8)
    plot.add(F, color="grey", alpha=0.2)
    
    if highlight is not None:
        for idx in highlight:
            plot.add(F[int(idx)], linewidth=3, color="red")
            
    plot.show()