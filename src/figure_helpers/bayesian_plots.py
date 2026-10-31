"""Manuscript definitions collected from build_position_pair_report.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import numpy as np

from matplotlib.colors import Normalize

from maze_analysis.plotting import (configure_matplotlib, style_axis, add_project_colorbar,
    output_config_tag, tagged_output_name, plot_day_level_group, GROUP_COLORS)

PLOT_GROUPS=('C','P')

LABELS={'accuracy':'Exact bin accuracy (%)','balanced_accuracy':'Mean bin recall (%)',
        'path_mean_cm':'Mean path error (cm)','euclidean_mean_cm':'decoding error'}

def heatmap(ax,matrix,title,vmax,xlabel):
    assert np.nanmax(matrix)<=vmax
    im=ax.imshow(matrix,origin='lower',interpolation='nearest',aspect='equal',
                 cmap='magma',norm=Normalize(0,vmax))
    ax.set(title=title,xlabel=xlabel,ylabel='True position bin',xticks=[0,14,56,97],yticks=[0,14,56,97])
    for b in (13.5,55.5):
        ax.axhline(b,color='.65',lw=.5,ls=':');ax.axvline(b,color='.65',lw=.5,ls=':')
    return im

def ceiling(values,minimum):
    return max(minimum,float(np.ceil(np.nanmax(values)*20)/20))

def performance(ax,metric,long,summary,days,model='Bayes'):
    scale=100 if metric in ('accuracy','balanced_accuracy') else 1
    obs=long[(long.model==model)&(long.metric==metric)].copy();obs['value']*=scale
    stat=summary[(summary.model==model)&(summary.metric==metric)].copy();stat[['mean','sem']]*=scale
    for g in PLOT_GROUPS:
        plot_day_level_group(ax,stat,obs,group=g,unit_col='mouse',error_style='band',individual_display='none')
        if model=='Bayes':
            null=summary[(summary.model=='Shift control')&(summary.metric==metric)&(summary.group==g)].sort_values('day')
            ax.plot(null.day,null['mean']*scale,c=GROUP_COLORS[g],ls='--',lw=1)
    ax.set(xlabel='Day',ylabel=LABELS[metric],xticks=days);style_axis(ax)
