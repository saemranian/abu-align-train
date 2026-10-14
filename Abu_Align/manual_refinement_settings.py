"""Validated settings and accuracy guidance for both camera families."""
import math

DEFAULTS={'loss':'soft_l1','loss_scale':1.,'auto_sigma_px':1.,'min_track_views':2,'min_angle_deg':1.,'ftol':1e-8,'xtol':1e-8,'gtol':1e-8}
GUIDES={
 'backend':'Auto uses CUDA for larger jobs and analytic CPU for small ones. CUDA changes speed, not measurement accuracy. Choose CUDA for large scenes with sufficient VRAM; Auto can fall back after GPU memory exhaustion.',
 'max_points':'Default 10000. Increase to 25000–50000 for more good multi-view support, at greater time/RAM/VRAM cost. Decrease for faster trials, possibly weaker coverage. Weak/wrong tracks can worsen accuracy. Manual tie/GCPs remain included. Untracked VGGT cloud points cannot become corrected tracks; their source is retained.',
 'iterations':'Default 100. Increase to 200–500 if the report says the budget was reached while still improving. Decrease for quicker trials that may stop early. CUDA counts LM steps; CPU counts residual evaluations. A larger limit does not cure wrong measurements/calibration.',
 'sigma_px':'New-click uncertainty in original pixels; use about 0.5–1 for a crisp accurately marked corner, 2–5 for blurred/stitching-sensitive features. Existing measurements keep their own uncertainty. Smaller values increase influence, not actual accuracy.',
 'loss':'soft_l1 is a smooth robust default; huber also reduces outlier influence. linear is ordinary least squares and suits clean verified observations. Robust loss cannot make a wrong correspondence correct; inspect high residuals.',
 'loss_scale':'Default 1, matching SciPy f_scale (normalized σ units). Increase to keep moderate discrepancies more influential; decrease to downweight them sooner. Too high trusts outliers; too low weakens useful constraints. Tune real uncertainty first and judge independent checks.',
 'auto_sigma_px':'Default 1 original pixel. Increase to reduce noisy automatic tracks\' influence; decrease to trust them more. Too small can let bad tracks overpower verified controls. Choose real track noise and compare independent checks; do not invent tiny manual/GCP uncertainty.',
 'min_track_views':'Automatic landmarks must occur in this many registered images. 3–4 views can improve track reliability if enough coverage remains; 2 keeps sparse/limited-overlap scenes connected. This does not remove enabled manual landmarks.',
 'min_angle_deg':'Default 1°, matching OpenSfM triangulation_min_ray_angle. Increase to reject weak depth geometry; decrease for distant/low-baseline scenes and keep more tracks. Too high disconnects cameras; too low retains unstable depth. 0 disables this extra filter. Manual landmarks remain included.',
 'ftol':'Default 1e-8. Decrease to require smaller relative objective changes before stopping, potentially more work; increase to stop sooner. Tighter values can exhaust the budget. They do not improve wrong observations. Inspect convergence and independent checks.',
 'xtol':'Default 1e-8. Decrease to require finer parameter steps, potentially more work; increase to stop sooner. Very tight values can exhaust the budget. Scene scale matters; correct calibration and controls matter more than tiny tolerances.',
 'gtol':'Default 1e-8, matching SciPy. Decrease to demand a smaller gradient and potentially more work; increase to stop earlier. CPU and CUDA definitions differ. Tighter tolerances cannot correct wrong points/calibration and can hit the iteration limit. Judge held-out checks.',
 'marker_size':'Default 6 screen pixels, an Abu display setting. Increase for visibility; decrease to uncover image detail. It changes marker appearance only, never measurements or optimization accuracy.',
 'survey':'Survey σ is the real uncertainty in metres, not a desired result. Use measured survey precision; check points are withheld from optimization. Distribute GCPs and checks through the scene and include depth/height variation.',
 'distance':'Distance and σ are in metres. Use independently measured endpoints and realistic precision. This establishes scale; it cannot replace distributed image observations or repair all camera errors.'}

ORIGINS={'loss':'Same robust family as installed OpenSfM SoftLOneLoss; SciPy default is linear. New OpenSfM master uses IRLS. These solvers and units differ.',
 'loss_scale':'SciPy f_scale = 1 in uncertainty-normalized units.',
 'min_track_views':'OpenSfM min_track_length = 2; Abu applies this to existing automatic tracks.',
 'min_angle_deg':'OpenSfM triangulation_min_ray_angle = 1 degree; Abu filters initial automatic landmark geometry.',
 'ftol':'SciPy ftol = 1e-8.','xtol':'SciPy xtol = 1e-8.','gtol':'SciPy gtol = 1e-8.',
 'iterations':'100 also matches OpenSfM bundle_max_iterations; CPU residual evaluations and CUDA LM steps differ.',
 'max_points':'10000 is an Abu resource budget, not an upstream accuracy standard.',
 'sigma_px':'1 original pixel is an Abu noise assumption; OpenSfM normalized reprojection SD 0.004 is not equivalent.',
 'auto_sigma_px':'1 original pixel is an Abu noise assumption; calibrate against your track quality.',
 'backend':'Auto is an Abu dispatch choice; hardware changes speed, not guaranteed accuracy.',
 'marker_size':'6 screen pixels is an Abu display choice.'}


def guide(key):return GUIDES[key]+'\n\nDefault source: '+ORIGINS.get(key,'Use the actual measured uncertainty.')


def validate(options=None):
    value=dict(DEFAULTS);value.update(options or {})
    if set(value)-set(DEFAULTS):raise ValueError('Unknown refinement setting.')
    if value['loss'] not in {'soft_l1','huber','linear'}:raise ValueError('Choose soft_l1, huber or linear loss.')
    for key in ['loss_scale','auto_sigma_px','ftol','xtol','gtol']:
        value[key]=float(value[key])
        if not math.isfinite(value[key]) or value[key]<=0:raise ValueError(f'{key} must be finite and positive.')
    for key in ['ftol','xtol','gtol']:
        if not 1e-12<=value[key]<=.01:raise ValueError(f'{key}: use 1e-12 to 0.01.')
    raw=float(value['min_track_views']);value['min_track_views']=int(raw)
    if raw!=value['min_track_views'] or not 2<=raw<=20:raise ValueError('Minimum automatic views must be an integer from 2 to 20.')
    value['min_angle_deg']=float(value['min_angle_deg'])
    if not math.isfinite(value['min_angle_deg']) or not 0<=value['min_angle_deg']<90:raise ValueError('Minimum parallax: use 0 to below 90 degrees.')
    return value
