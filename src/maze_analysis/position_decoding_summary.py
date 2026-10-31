"""Manuscript definitions collected from position_decoding_summary.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import json

import numpy as np

import pandas as pd

from .manuscript_lmm import fit_longitudinal_lmm

ERROR_METRICS = ('mae_cm', 'distance_cm')

def fit_position_inference(mouse_day, *, stat_groups=('C', 'I', 'P'),
                           stat_metrics=ERROR_METRICS, return_details=False):
    """Use manuscript REML/Wald contrasts, without covariates as requested.

    Every estimand comes from the same day-by-group mouse-intercept model.
    Effects/SE/CI are in mm (or mm/day); source metrics retain their cm keys.
    """
    if not set(stat_metrics).issubset(ERROR_METRICS):
        raise ValueError('position inference now includes MAE and Distance Error only')
    results, coefficients, diagnostics = [], [], []
    effect_types = {
        'reference_group_day_slope': 'within_group_day_slope',
        'comparison_group_day_slope': 'within_group_day_slope',
        'group_by_day_interaction': 'group_day_interaction',
        'centered_overall_group_effect': 'between_group_overall',
        'between_group_D1': 'between_group_first_day',
        'between_group_D5': 'between_group_last_day',
    }
    for comparison in ('I', 'P'):
        if 'C' not in stat_groups or comparison not in stat_groups:
            continue
        for view in ('real', 'real_minus_shuffle'):
            for metric in stat_metrics:
                table = mouse_day.loc[(mouse_day['view'] == view) & (mouse_day['metric'] == metric)
                                     & mouse_day.group.isin(('C',comparison))].copy()
                table['value_mm'] = table.value * 10
                fit = fit_longitudinal_lmm(table,value_col='value_mm',reference_group='C',
                    comparison_group=comparison,continuous_covariates=(),
                    sampling_unit_cols=('mouse','day'),first_time=1,last_time=5)
                meta = dict(analysis='position_decoding',view=view,metric=metric,contrast=f'{comparison}-C')
                frame = fit.contrasts.rename(columns={'contrast':'manuscript_contrast',
                    'beta':'effect','z_value':'statistic','n_mice':'n_mouse','nobs':'n_mouse_day'}).copy()
                for key,value in meta.items():
                    frame[key] = value
                frame['effect_type'] = frame.manuscript_contrast.map(effect_types)
                if frame.effect_type.isna().any():
                    raise ValueError('unrecognized manuscript LMM contrast')
                slopes = frame.effect_type.isin(('within_group_day_slope','group_day_interaction'))
                frame['effect_unit'] = np.where(slopes,'mm/day','mm')
                frame['sampling_unit'] = 'mouse'
                frame['model_role'] = 'unadjusted_by_user_request'
                frame['multiplicity_correction'] = 'none'
                frame['reported_probability'] = 'P'
                valid = np.isfinite(frame.p_value) & np.isfinite(frame.standard_error) & (frame.standard_error>0)
                frame['estimability'] = np.where(valid,'estimable','n.e.')
                frame['significant'] = pd.array(np.where(valid,frame.p_value<.05,None),dtype='boolean')
                results.append(frame)
                coefs = fit.coefficients.copy()
                for key,value in meta.items():
                    coefs[key] = value
                coefficients.append(coefs)
                diagnostic = dict(**fit.diagnostics,**meta)
                diagnostic = {k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v
                              for k,v in diagnostic.items()}
                diagnostics.append(pd.DataFrame([diagnostic]))
    combined = pd.concat(results,ignore_index=True) if results else pd.DataFrame()
    if not return_details:
        return combined
    return dict(inference=combined,
                coefficients=pd.concat(coefficients,ignore_index=True) if coefficients else pd.DataFrame(),
                diagnostics=pd.concat(diagnostics,ignore_index=True) if diagnostics else pd.DataFrame())
