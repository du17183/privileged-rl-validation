"""Paired conditional slices and isolated perturbations, nominal physics only."""
CONDITIONS={
 'nominal':(0,None), 'level2':(2,None),
 'angle_zero':(2,{'angle_deg':0.}),
 'angle_low':(2,{'angle_range_deg':(0.,2.5)}),
 'angle_high':(2,{'angle_range_deg':(2.5,5.)}),
 'position_zero':(2,{'offset_xyz':(0.,0.,0.)}),
 'position_halfcm':(2,{'offset_bound':.005}),
 'position_onecm':(2,{'offset_bound':.01}),
 # Shells make small/large position effects distinguishable from nested cubes.
 'position_shell_halfcm':(2,{'offset_linf_range':(.0025,.005)}),
 'position_shell_onecm':(2,{'offset_linf_range':(.005,.01)}),
}
