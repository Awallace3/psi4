"""Closed selections for shared descriptor engines, not runtime admission."""


def profile(producer='cb63305', *, scope='default'):
    """Return fresh configuration data; never accept modules or arbitrary scope."""
    if type(producer) is not str or producer not in ('cb63305', 'ed561666', 'grid400', 'qomp'):
        raise ValueError('closed installed descriptor producer required')
    if (type(scope) is not str or scope not in ('default', 'full-bank')
            or scope == 'full-bank' and producer not in ('grid400', 'qomp')):
        raise ValueError('closed grid400 full-bank descriptor scope required')
    children = ('descriptor_probe_cb63305.py', 'descriptor_probe.py',
                'resource_plan_cb63305.py', 'descriptor_profiles.py')
    extras = ()
    if producer == 'ed561666':
        children += ('resource_plan_ed561666.py',)
        extras = ('qualified_ed561666.py', 'resource_plan_ed561666.py',
                  'prepare_ed561666_inputs.py')
    elif producer == 'grid400':
        children += ('resource_plan_ed561666.py', 'resource_plan_grid400.py')
        extras = ('qualified_grid400.py', 'resource_plan_ed561666.py',
                  'resource_plan_grid400.py')
    elif producer == 'qomp':
        children += ('resource_plan_ed561666.py', 'resource_plan_grid400.py',
                     'resource_plan_isa_grid_v1.py', 'resource_plan_qomp.py',
                     'qualified_qomp.py', 'runtime_qomp.py', 'qomp_bank_policy.py',
                     'bank_runtime_gate.py', 'grid400_bank_policy.py', 'runtime_audit.py',
                     'descriptor_scratch_cb63305.py')
        extras = ('qualified_qomp.py', 'runtime_qomp.py', 'resource_plan_ed561666.py',
                  'resource_plan_grid400.py', 'resource_plan_isa_grid_v1.py', 'resource_plan_qomp.py',
                  'qomp_bank_policy.py', 'bank_runtime_gate.py', 'grid400_bank_policy.py',
                  'runtime_audit.py', 'descriptor_scratch_cb63305.py')
    return dict(producer=producer, qualifier='qualified_'+producer,
        planner='resource_plan_'+producer, child_sources=children, parent_extra_sources=extras,
        drho_refinement_iterations=10 if producer == 'cb63305' else 32,
        integration_grid=(400, 2030) if producer in ('grid400', 'qomp') else (200, 974),
        selected_monomers=('monomer-0008',)
            if producer == 'grid400' and scope == 'default' else None)
