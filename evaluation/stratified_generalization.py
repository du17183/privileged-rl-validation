"""Fixed physical strata; no illegal negative angle and no silent clipping."""
CONDITIONS = {
    'nominal': (0, None),
    'level1': (1, None),
    'level2': (2, None),
    'level3': (3, None),
    'angle_0_1': (0, {'angle_range_deg': [0., 1.]}),
    'angle_1_2p5': (0, {'angle_range_deg': [1., 2.5]}),
    'angle_2p5_5': (0, {'angle_range_deg': [2.5, 5.]}),
    'angle_2p5': (0, {'angle_deg': 2.5}),
    'angle_5': (0, {'angle_deg': 5.}),
    'offset_0p005': (0, {'offset_bound': .005}),
    'offset_0p01': (0, {'offset_bound': .01}),
    'offset_shell_0_0p25cm': (2, {'offset_linf_range': [0., .0025]}),
    'offset_shell_0p25_0p5cm': (2, {'offset_linf_range': [.0025, .005]}),
    'offset_shell_0p5_1cm': (2, {'offset_linf_range': [.005, .01]}),
    'offset_y_minus_0p01': (0, {'offset_xyz': [0., -.01, 0.]}),
    'offset_y_plus_0p01': (0, {'offset_xyz': [0., .01, 0.]}),
    'friction_0p9': (0, {'friction_scale': .9}),
    'friction_1p1': (0, {'friction_scale': 1.1}),
}


def slice_episodes(records):
    """Joint Level2 distribution, bins by initial angle and fixture Linf shift.

Other nuisance factors remain randomized in these conditional bins; report
counts and avoid confusing them with isolated condition tests.
"""
    import math
    results = {}
    for field, edges in (('initial_angle_deg', [(0., 1.), (1., 2.5), (2.5, 5.00001)]),
                         ('offset_linf_cm', [(0., .25), (.25, .5), (.5, 1.00001)])):
        for lower, upper in edges:
            rows = [r for r in records if lower <= r[field] < upper]
            results[f'{field}_{lower}_{upper}'] = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                success=sum(r['success'] for r in rows)/len(rows) if rows else None)
    return results
