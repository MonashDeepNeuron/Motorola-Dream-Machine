"""
Light-weight helpers that are *pure* (no printing, no plots) so they can be
imported by producer *or* consumer without side-effects.
"""

from .header import header_dump                 
from .bands import compute_window_band_power, DefaultBands     
from .events import event_band_statistics        
from .plotting import plot_avg_band_power, plot_band_power_over_time 