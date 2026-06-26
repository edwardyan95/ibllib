"""
Multimodality score analysis for Purkinje-cell dendrite topography.

Score definition used here:
    visual  = any significant visual stimulus predictor
    auditory = any significant auditory rule-cue or auditory stimulus predictor
    motion = any significant body/lick/face predictor
    score = visual + auditory + motion, range 0..3

Expected input:
    agg_data[mouse][date][arm]['stat'] contains neuron dictionaries with
        - 'unique_explained_variance'
        - 'bootstrap_p_value'
        - 'xcoord_atlas'
        - 'ycoord_atlas'
    agg_data[mouse]['training'] contains
        - 'accuracies_vis'
        - 'accuracies_aud'
        - 'training_dates'

This file is designed to be pasted into, or imported from, the existing analysis notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


UNIQUE_PREDICTORS_BASE = [
    'vis_ruleCue_times',
    'aud_ruleCue_times',
    'left_vis_stimOn_times',
    'right_vis_stimOn_times',
    'left_aud_stimOn_times',
    'right_aud_stimOn_times',
    'vis_left_choice_times',
    'vis_right_choice_times',
    'aud_left_choice_times',
    'aud_right_choice_times',
    'vis_reward_times',
    'aud_reward_times',
    'vis_punish_times',
    'aud_punish_times',
    'omission_times',
    'previous_feedbackType',
    'reward_history',
    'body',
    'lick',
    'face',
    'pupil_area',
    'pupil_xpos',
    'pupil_ypos',
    'vel',
]

# Main 3-category definition.
# Note: vis_ruleCue_times and aud_ruleCue_times are auditory sounds, so they are
# treated as auditory modality, not visual modality.
MODALITY_GROUPS_VAM = {
    'visual': [
        'left_vis_stimOn_times',
        'right_vis_stimOn_times',
    ],
    'auditory': [
        'vis_ruleCue_times',
        'aud_ruleCue_times',
        'left_aud_stimOn_times',
        'right_aud_stimOn_times',
    ],
    'motion': [
        'body',
        'lick',
        'face',
    ],
}

STAGE_ORDER = ['naive', 'vis_expert', 'aud_expert', 'db_expert']
STAGE_TITLES = {
    'naive': 'Naive',
    'vis_expert': 'Vis expert',
    'aud_expert': 'Aud expert',
    'db_expert': 'Double expert',
}


def _as_str_array(x):
    """Convert date arrays that may be bytes/ints/strings into string arrays."""
    out = []
    for v in np.asarray(x):
        if isinstance(v, bytes):
            out.append(v.decode())
        else:
            out.append(str(v))
    return np.asarray(out)


def rolling_average(x, w=3):
    x = np.asarray(x, dtype=float)
    if w <= 1 or x.size == 0:
        return x
    pad = np.pad(x, (w - 1, 0), mode='edge')
    c = np.cumsum(pad)
    out = (c[w:] - c[:-w]) / w
    head = np.full(w - 1, out[0])
    return np.concatenate([head, out])


def classify_day_stage(vis_perf, aud_perf, vis_raw, aud_raw, perf_thresh=0.65):
    """Return one of: naive, vis_expert, aud_expert, db_expert."""
    v = float(vis_perf) if np.isfinite(vis_perf) else float(vis_raw)
    a = float(aud_perf) if np.isfinite(aud_perf) else float(aud_raw)
    v_ok = (v > perf_thresh) or (float(vis_raw) > perf_thresh)
    a_ok = (a > perf_thresh) or (float(aud_raw) > perf_thresh)
    if v_ok and a_ok:
        return 'db_expert'
    if v_ok and not a_ok:
        return 'vis_expert'
    if a_ok and not v_ok:
        return 'aud_expert'
    return 'naive'


def is_imaging_date_key(s):
    """Match session keys used in agg_data."""
    if not isinstance(s, str):
        return False
    if s.isdigit():
        return True
    suffix1 = '_valvesilent'
    suffix2 = '_valvesilent_punishsilent'
    return (s.endswith(suffix1) and s[:-len(suffix1)].isdigit()) or (
        s.endswith(suffix2) and s[:-len(suffix2)].isdigit()
    )


def _get_pred_names(arm_data, unique_predictors_base=None):
    """
    Prefer unique_predictors_extended if saved in arm_data; otherwise fall back to
    the passed base list. The V/A/M score only uses base predictors, so this also
    works if choice_group predictors were appended to the arrays but not to the names.
    """
    if 'unique_predictors_extended' in arm_data:
        return list(arm_data['unique_predictors_extended'])
    if 'unique_predictors' in arm_data:
        return list(arm_data['unique_predictors'])
    if unique_predictors_base is None:
        return list(UNIQUE_PREDICTORS_BASE)
    return list(unique_predictors_base)


def _valid_xy(neuron, atlas_shape=None):
    if ('xcoord_atlas' not in neuron) or ('ycoord_atlas' not in neuron):
        return None, None
    x = np.asarray(neuron['xcoord_atlas']).ravel()
    y = np.asarray(neuron['ycoord_atlas']).ravel()
    if x.size == 0 or y.size == 0:
        return None, None
    if atlas_shape is not None:
        H, W = map(int, atlas_shape)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not np.any(ok):
            return None, None
        x = x[ok]
        y = y[ok]
    return x, y


def modality_group_stats(
    neuron,
    pred_names,
    group_predictors,
    p_thresh=0.01,
    min_uev=0.0,
    within_category_bonferroni=False,
):
    """
    Compute a category-level significance flag.

    The requested rule is: at least one predictor in the category must be significant.
    By default, significance means p < p_thresh and UEV > min_uev for any predictor.

    If within_category_bonferroni=True, use p_thresh / number_of_available_predictors
    inside the category. This is a sensitivity option because auditory has more predictors
    than visual.
    """
    pvals = np.asarray(neuron.get('bootstrap_p_value', []), dtype=float)
    uev = np.asarray(neuron.get('unique_explained_variance', []), dtype=float)

    idxs = []
    names = []
    for p in group_predictors:
        if p in pred_names:
            idx = pred_names.index(p)
            if idx < pvals.size and idx < uev.size:
                idxs.append(idx)
                names.append(p)

    if len(idxs) == 0:
        return {
            'flag': False,
            'p_min': np.nan,
            'p_group_bonf': np.nan,
            'uev_max': np.nan,
            'uev_sum_pos': np.nan,
            'drivers': [],
            'available_predictors': [],
            'threshold_used': np.nan,
        }

    p_group = pvals[idxs]
    uev_group = uev[idxs]
    thresh_used = p_thresh / len(idxs) if within_category_bonferroni else p_thresh
    sig_mask = (p_group < thresh_used) & (uev_group > min_uev)

    drivers = [nm for nm, ok in zip(names, sig_mask) if bool(ok)]
    uev_pos = np.maximum(uev_group, 0)

    return {
        'flag': bool(np.any(sig_mask)),
        'p_min': float(np.nanmin(p_group)),
        'p_group_bonf': float(min(np.nanmin(p_group) * len(idxs), 1.0)),
        'uev_max': float(np.nanmax(uev_pos)),
        'uev_sum_pos': float(np.nansum(uev_pos)),
        'drivers': drivers,
        'available_predictors': names,
        'threshold_used': float(thresh_used),
    }


def score_neuron_vam(
    neuron,
    pred_names,
    modality_groups=None,
    p_thresh=0.01,
    min_uev=0.0,
    within_category_bonferroni=False,
    store=True,
    key_suffix='vam',
):
    """
    Score one dendrite for visual/auditory/motion multimodality.

    Returns
    -------
    result : dict
        Contains flags, score, label, p minima, UEV summaries, and drivers.
    """
    if modality_groups is None:
        modality_groups = MODALITY_GROUPS_VAM

    result = {
        'flags': {},
        'p_min': {},
        'p_group_bonf': {},
        'uev_max': {},
        'uev_sum_pos': {},
        'drivers': {},
        'available_predictors': {},
        'threshold_used': {},
    }

    for modality, predictors in modality_groups.items():
        stats = modality_group_stats(
            neuron,
            pred_names,
            predictors,
            p_thresh=p_thresh,
            min_uev=min_uev,
            within_category_bonferroni=within_category_bonferroni,
        )
        result['flags'][modality] = stats['flag']
        result['p_min'][modality] = stats['p_min']
        result['p_group_bonf'][modality] = stats['p_group_bonf']
        result['uev_max'][modality] = stats['uev_max']
        result['uev_sum_pos'][modality] = stats['uev_sum_pos']
        result['drivers'][modality] = stats['drivers']
        result['available_predictors'][modality] = stats['available_predictors']
        result['threshold_used'][modality] = stats['threshold_used']

    score = int(sum(bool(v) for v in result['flags'].values()))
    label_parts = [m for m in modality_groups.keys() if result['flags'][m]]
    label = '+'.join(label_parts) if label_parts else 'none'

    result['score'] = score
    result['label'] = label

    if store:
        neuron[f'multimodal_score_{key_suffix}'] = score
        neuron[f'multimodal_flags_{key_suffix}'] = dict(result['flags'])
        neuron[f'multimodal_label_{key_suffix}'] = label
        neuron[f'multimodal_drivers_{key_suffix}'] = dict(result['drivers'])
        neuron[f'multimodal_uevmax_{key_suffix}'] = dict(result['uev_max'])
        neuron[f'multimodal_pmin_{key_suffix}'] = dict(result['p_min'])

    return result


def rasterize_dendrites_full_atlas(
    neurons,
    atlas_shape,
    downsample=4,
    min_count=10,
    smooth_sigma_bins=1.0,
    weighting='none',
):
    """
    Rasterize dendrite values over the full atlas field.

    weighting options:
        'length': pixel-count weighted, matching the original UEV maps most closely.
        'none'  : each dendrite contributes one vote to each downsampled bin it touches.
        'unit'  : each dendrite contributes total weight 1 divided across all touched bins.

    For multimodality scores, 'none' is a good default because the unit of analysis is
    the dendrite, not dendritic pixel count.
    """
    H, W = map(int, atlas_shape)
    nx_ds = (W + downsample - 1) // downsample
    ny_ds = (H + downsample - 1) // downsample
    nb = nx_ds * ny_ds

    H_count_flat = np.zeros(nb, dtype=np.float32)
    H_sum_flat = np.zeros(nb, dtype=np.float32)
    inv_ds = 1.0 / float(downsample)

    for neu in neurons:
        x = np.asarray(neu['xcoord_atlas']).ravel().astype(np.int64)
        y = np.asarray(neu['ycoord_atlas']).ravel().astype(np.int64)
        d = float(neu['data'])
        if x.size == 0 or y.size == 0 or not np.isfinite(d):
            continue

        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not np.any(ok):
            continue
        x = x[ok]
        y = y[ok]

        ix_ds = (x * inv_ds).astype(np.int64)
        iy_ds = (y * inv_ds).astype(np.int64)
        lin = iy_ds * nx_ds + ix_ds
        u, c = np.unique(lin, return_counts=True)

        if weighting == 'unit':
            total = c.sum()
            wcounts = (c / total).astype(np.float32) if total > 0 else np.zeros_like(c, dtype=np.float32)
        elif weighting == 'length':
            wcounts = c.astype(np.float32)
        elif weighting == 'none':
            wcounts = np.ones_like(c, dtype=np.float32)
        else:
            raise ValueError("weighting must be 'length', 'none', or 'unit'")

        H_count_flat[u] += wcounts
        H_sum_flat[u] += wcounts * d

    with np.errstate(invalid='ignore', divide='ignore'):
        H_mean_flat = H_sum_flat / H_count_flat

    H_count = H_count_flat.reshape(ny_ds, nx_ds)
    H_mean = H_mean_flat.reshape(ny_ds, nx_ds)
    H_mean[H_count < min_count] = np.nan

    if smooth_sigma_bins and smooth_sigma_bins > 0:
        mask = np.isfinite(H_mean)
        Hf = np.where(mask, H_mean, 0.0)
        wf = gaussian_filter(mask.astype(float), smooth_sigma_bins, mode='nearest')
        hf = gaussian_filter(Hf, smooth_sigma_bins, mode='nearest')
        H_mean = np.where(wf > 1e-9, hf / wf, np.nan)

    extent = [0, W, 0, H]
    return H_mean, H_count, extent


def assign_neuron_to_roi(neuron, roi_masks_bool, frac_thresh=0.2):
    """Optional ROI assignment; compatible with the existing ROI mask logic."""
    x, y = _valid_xy(neuron)
    if x is None:
        return None, 0.0

    x = x.astype(int)
    y = y.astype(int)
    best_roi = None
    best_frac = 0.0
    for roi_name, roi_mask in roi_masks_bool.items():
        H, W = roi_mask.shape
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not np.any(ok):
            continue
        xx = x[ok]
        yy = y[ok]
        frac = np.count_nonzero(roi_mask[yy, xx]) / xx.size
        if frac > best_frac:
            best_frac = float(frac)
            best_roi = roi_name

    if best_roi is not None and best_frac > frac_thresh:
        return best_roi, best_frac
    return None, 0.0


def collect_multimodality_score_neurons(
    agg_data,
    mouse_ids,
    atlas_shape=None,
    unique_predictors_base=None,
    modality_groups=None,
    perf_thresh=0.65,
    p_thresh=0.01,
    min_uev=0.0,
    within_category_bonferroni=False,
    key_suffix='vam',
    roi_masks_full=None,
    roi_frac_thresh=0.2,
):
    """
    Score all dendrites and return raster-ready pools plus a tidy DataFrame.

    Returns
    -------
    pooled : dict stage -> list of {'xcoord_atlas', 'ycoord_atlas', 'data'}
        data is the multimodality score 0..3.
    df : pandas.DataFrame
        One row per dendrite with score, flags, p minima, UEV summaries, and drivers.
    """
    if unique_predictors_base is None:
        unique_predictors_base = UNIQUE_PREDICTORS_BASE
    if modality_groups is None:
        modality_groups = MODALITY_GROUPS_VAM

    pooled = {stage: [] for stage in STAGE_ORDER}
    records = []

    roi_masks_bool = None
    if roi_masks_full is not None:
        roi_masks_bool = {k: np.asarray(v, dtype=bool) for k, v in roi_masks_full.items()}

    for mouse in mouse_ids:
        md = agg_data[mouse]
        vis_acc = np.asarray(md['training']['accuracies_vis'], dtype=float)
        aud_acc = np.asarray(md['training']['accuracies_aud'], dtype=float)
        tr_dates = _as_str_array(md['training']['training_dates'])
        vis_roll = rolling_average(vis_acc, w=3)
        aud_roll = rolling_average(aud_acc, w=3)

        img_dates = [s for s in md.keys() if is_imaging_date_key(s)]

        for date in img_dates:
            date8 = date[:8]
            idx = np.where(tr_dates == date8)[0]
            if idx.size == 0:
                continue
            di = int(idx[0])
            stage = classify_day_stage(vis_roll[di], aud_roll[di], vis_acc[di], aud_acc[di], perf_thresh)

            for arm in ['arm1', 'arm2']:
                if arm not in md[date]:
                    continue
                arm_data = md[date][arm]
                pred_names = _get_pred_names(arm_data, unique_predictors_base=unique_predictors_base)

                for ni, neuron in enumerate(arm_data['stat']):
                    x, y = _valid_xy(neuron, atlas_shape=atlas_shape)
                    if x is None:
                        continue

                    res = score_neuron_vam(
                        neuron,
                        pred_names,
                        modality_groups=modality_groups,
                        p_thresh=p_thresh,
                        min_uev=min_uev,
                        within_category_bonferroni=within_category_bonferroni,
                        store=True,
                        key_suffix=key_suffix,
                    )

                    pooled[stage].append({
                        'xcoord_atlas': x,
                        'ycoord_atlas': y,
                        'data': float(res['score']),
                    })

                    roi_name = neuron.get('roi', neuron.get('roi_name', None))
                    roi_frac = np.nan
                    if roi_masks_bool is not None:
                        roi_name, roi_frac = assign_neuron_to_roi(
                            neuron,
                            roi_masks_bool,
                            frac_thresh=roi_frac_thresh,
                        )

                    rec = {
                        'mouse': mouse,
                        'date': date,
                        'date8': date8,
                        'arm': arm,
                        'neuron_idx': ni,
                        'stage': stage,
                        'score': int(res['score']),
                        'label': res['label'],
                        'n_atlas_pixels': int(len(x)),
                        'roi': roi_name,
                        'roi_frac': roi_frac,
                    }
                    for modality in modality_groups.keys():
                        rec[f'{modality}_sig'] = bool(res['flags'][modality])
                        rec[f'{modality}_pmin'] = res['p_min'][modality]
                        rec[f'{modality}_pgroup_bonf'] = res['p_group_bonf'][modality]
                        rec[f'{modality}_uevmax'] = res['uev_max'][modality]
                        rec[f'{modality}_uevsum_pos'] = res['uev_sum_pos'][modality]
                        rec[f'{modality}_drivers'] = '|'.join(res['drivers'][modality])
                    records.append(rec)

    df = pd.DataFrame.from_records(records)
    return pooled, df


def build_stage_maps_from_pooled(
    pooled,
    atlas_shape,
    downsample=4,
    min_count=10,
    smooth_sigma_bins=1.0,
    weighting='none',
):
    stage_maps = {}
    stage_count_maps = {}
    extent = [0, atlas_shape[1], 0, atlas_shape[0]]
    for stage in STAGE_ORDER:
        neurons = pooled.get(stage, [])
        if len(neurons) == 0:
            stage_maps[stage] = None
            stage_count_maps[stage] = None
            continue
        H_mean, H_count, extent = rasterize_dendrites_full_atlas(
            neurons,
            atlas_shape=atlas_shape,
            downsample=downsample,
            min_count=min_count,
            smooth_sigma_bins=smooth_sigma_bins,
            weighting=weighting,
        )
        stage_maps[stage] = H_mean
        stage_count_maps[stage] = H_count
    return stage_maps, stage_count_maps, extent


def plot_multimodality_stage_maps(
    stage_maps,
    atlas_template,
    extent=None,
    vmin=0,
    vmax=3,
    cmap='viridis',
    alpha=0.9,
    title='Mean multimodality score',
):
    H, W = atlas_template.shape
    if extent is None:
        extent = [0, W, 0, H]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    im_last = None
    for ax, stage in zip(axes.ravel(), STAGE_ORDER):
        ax.imshow(atlas_template, cmap='gray', extent=[0, W, 0, H], origin='upper')
        Hm = stage_maps.get(stage, None)
        if Hm is not None:
            im_last = ax.imshow(
                Hm,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin='upper',
                alpha=alpha,
                interpolation='nearest',
            )
        ax.set_title(STAGE_TITLES[stage], fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

    if im_last is not None:
        cbar = fig.colorbar(im_last, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label(title, rotation=90)
        cbar.set_ticks([0, 1, 2, 3])
    return fig, axes


def make_multimodality_score_stage_maps(
    agg_data,
    mouse_ids,
    atlas_template,
    unique_predictors_base=None,
    modality_groups=None,
    perf_thresh=0.65,
    p_thresh=0.01,
    min_uev=0.0,
    within_category_bonferroni=False,
    downsample=4,
    min_count=10,
    smooth_sigma_bins=1.0,
    weighting='none',
    key_suffix='vam',
    roi_masks_full=None,
    roi_frac_thresh=0.2,
    plot=True,
):
    """
    End-to-end wrapper for visual/auditory/motion multimodality score maps.
    """
    pooled, df = collect_multimodality_score_neurons(
        agg_data=agg_data,
        mouse_ids=mouse_ids,
        atlas_shape=atlas_template.shape,
        unique_predictors_base=unique_predictors_base,
        modality_groups=modality_groups,
        perf_thresh=perf_thresh,
        p_thresh=p_thresh,
        min_uev=min_uev,
        within_category_bonferroni=within_category_bonferroni,
        key_suffix=key_suffix,
        roi_masks_full=roi_masks_full,
        roi_frac_thresh=roi_frac_thresh,
    )
    stage_maps, stage_count_maps, extent = build_stage_maps_from_pooled(
        pooled=pooled,
        atlas_shape=atlas_template.shape,
        downsample=downsample,
        min_count=min_count,
        smooth_sigma_bins=smooth_sigma_bins,
        weighting=weighting,
    )
    fig = axes = None
    if plot:
        fig, axes = plot_multimodality_stage_maps(
            stage_maps=stage_maps,
            atlas_template=atlas_template,
            extent=extent,
            vmin=0,
            vmax=3,
            title='Mean V/A/M modality count per dendrite',
        )
    return fig, axes, stage_maps, stage_count_maps, extent, pooled, df


def summarize_multimodality(df, by='stage'):
    """Return score and label count/proportion tables."""
    if df.empty:
        return {}
    if isinstance(by, str):
        by_cols = [by]
    else:
        by_cols = list(by)

    score_counts = pd.crosstab([df[c] for c in by_cols], df['score']).reindex(columns=[0, 1, 2, 3], fill_value=0)
    score_props = score_counts.div(score_counts.sum(axis=1), axis=0)
    label_counts = pd.crosstab([df[c] for c in by_cols], df['label'])
    label_props = label_counts.div(label_counts.sum(axis=1), axis=0)
    return {
        'score_counts': score_counts,
        'score_props': score_props,
        'label_counts': label_counts,
        'label_props': label_props,
    }


def plot_score_proportions(df, by='stage'):
    """Quick stacked bar plot of score proportions."""
    tables = summarize_multimodality(df, by=by)
    props = tables['score_props']
    fig, ax = plt.subplots(figsize=(7, 4))
    props.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel('Fraction of dendrites')
    ax.set_xlabel(by if isinstance(by, str) else ' / '.join(by))
    ax.set_ylim(0, 1)
    ax.legend(title='Score', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    return fig, ax, props
