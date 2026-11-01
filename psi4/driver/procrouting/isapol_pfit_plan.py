# Psi4 Developers; SPDX-License-Identifier: LGPL-3.0-only
"""Allocation-free PFIT accounting shared by preflight and execution.

These are conservative numeric-buffer and declared-work policies, not total
process RSS, vendor runtime or a scientific solvability certificate. The native
kernel independently enforces its cap, including queried LAPACK workspace.
"""
import struct

MAX_PARAMETERS = 4096
MAX_REPLAY_WORK = 2_000_000_000_000
DEFAULT_REPLAY_WORK = 64_000_000_000
DEFAULT_NATIVE_WORK_BYTES = 256*1024**2
MAX_NATIVE_WORK_BYTES = (1 << (8*struct.calcsize('P')))-1
MAX_CHUNK = 64


def _integer(name, value, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be a positive integer'
                         + ('' if maximum is None else f' <= {maximum}'))
    return value


def check_replay_work(value):
    if type(value) is not int or not 1 <= value <= MAX_REPLAY_WORK:
        raise ValueError(f'max_work must be an int in 1..{MAX_REPLAY_WORK}')
    return value


def native_work_limit(value=None):
    """None retains the native256MiB default; explicit values are never raised."""
    if value is None:
        return DEFAULT_NATIVE_WORK_BYTES
    return _integer('native PFIT workspace budget', value, MAX_NATIVE_WORK_BYTES)


def packed_design_plan(*, npoint, channel_count, parameter_count, used_channels,
                       block_rows=256, max_passes=2, max_bytes=512*1024**2,
                       max_work=None, max_chunk=MAX_CHUNK):
    for name, value in (('npoint', npoint), ('channel_count', channel_count),
                        ('parameter_count', parameter_count), ('used_channels', used_channels)):
        _integer(name, value)
    if used_channels > channel_count or parameter_count > MAX_PARAMETERS:
        raise ValueError('invalid packed design channel/parameter dimensions')
    _integer('block_rows', block_rows, 4096)
    _integer('max_passes', max_passes, 2)
    _integer('max_bytes', max_bytes)
    _integer('max_chunk', max_chunk, MAX_CHUNK)
    budget = check_replay_work(DEFAULT_REPLAY_WORK if max_work is None else max_work)
    rows = npoint*(npoint+1)//2
    block = min(block_rows, rows)
    chunk = max(1, min(max_chunk, 16*block//npoint))
    numeric = (9*npoint*channel_count+8*npoint*(used_channels*(parameter_count+1)+1)
               +max_passes*8*(npoint*chunk*parameter_count+block*(2*parameter_count+12))+256)
    work = max_passes*((rows+max_chunk*npoint)*2*used_channels*parameter_count
                       +rows*(4*parameter_count+8))
    if numeric > max_bytes:
        raise ValueError('packed design byte resource limit exceeded')
    if work > budget:
        raise ValueError('packed design replay work resource limit exceeded')
    return dict(row_count=rows, block_rows=block, chunk=chunk,
                planned_bytes=numeric, planned_work=work, max_work=budget)


def streamed_overhead(npoint, channels, frequencies, block_rows, *, computed_fields):
    for name, value in (('npoint', npoint), ('channels', channels), ('frequencies', frequencies)):
        _integer(name, value)
    _integer('block_rows', block_rows, 4096)
    if type(computed_fields) is not bool:
        raise ValueError('computed_fields must be explicit bool')
    block = min(block_rows, npoint*(npoint+1)//2)
    return 24*npoint+(8*npoint*channels if computed_fields else 0)+17*block*frequencies


def native_row_plan(npoint, parameters, frequencies, *, block_rows=4096,
                    maximum_work_bytes=None, solver='NormalEquationsDSYSV',
                    qr_chunk_rows=256, retain_pair_predictions=False):
    """Exact source-derived native base for the all-free one-cloud row wrapper.

    Native capacity uses the declared block_rows, NOT min(block_rows, rows).
    All retained predictions and the live QR column-copy buffer are included.
    Passing the base gate does not guarantee later queried-workspace admission.
    """
    _integer('npoint', npoint)
    _integer('parameters', parameters, MAX_PARAMETERS)
    _integer('frequencies', frequencies, 2**31-1)
    _integer('block_rows', block_rows, 4096)
    _integer('qr_chunk_rows', qr_chunk_rows, 2**31-1)
    if solver not in ('NormalEquationsDSYSV', 'StreamingQR'):
        raise ValueError('unknown native PFIT solver')
    if type(retain_pair_predictions) is not bool:
        raise ValueError('retain_pair_predictions must be bool')
    cap = native_work_limit(maximum_work_bytes)
    rows = npoint*(npoint+1)//2
    source = 8*(block_rows*(parameters+frequencies)+parameters)
    each = 40*parameters**2+64*parameters+16
    if retain_pair_predictions:
        each += rows
    scalars = each*frequencies+block_rows*(parameters+2*frequencies)
    if solver == 'StreamingQR':
        scalars += qr_chunk_rows*(parameters+frequencies)+qr_chunk_rows*(parameters+1)
    base = 8*scalars+source
    if base > MAX_NATIVE_WORK_BYTES:
        raise ValueError('native PFIT workspace plan exceeds native representation')
    if base > cap:
        raise ValueError('native PFIT workspace budget (maximum_work_bytes) below kernel base plan')
    return dict(native_base_bytes=base, native_work_bytes=cap,
                native_source_bytes=source, native_block_rows=block_rows,
                solver=solver, queried_workspace='independently enforced at execution')


def fit_envelope(npoint, channels, parameters, frequencies, *, block_rows=4096,
                 max_chunk=MAX_CHUNK):
    """Original conservative fit envelope/work; native cap is reserved separately."""
    for name, value in (('npoint', npoint), ('channels', channels), ('frequencies', frequencies)):
        _integer(name, value)
    _integer('parameters', parameters, MAX_PARAMETERS)
    _integer('block_rows', block_rows, 4096)
    _integer('max_chunk', max_chunk, MAX_CHUNK)
    rows = npoint*(npoint+1)//2
    numeric = (8*(5*npoint*channels+npoint*channels*parameters+2*max_chunk*npoint*parameters
                  +16*parameters**2*frequencies+block_rows*(8*parameters+32*frequencies))
               +8*1024**2)
    work = (2*((rows+max_chunk*npoint)*2*channels*parameters+rows*(4*parameters+8))
            +2*rows*parameters**2+4*rows*parameters*frequencies
            +frequencies*(64*parameters**3+64*parameters**2))
    return numeric, work


def combined_fit_plan(npoint, channels, parameters, frequencies, *, block_rows=4096,
                      maximum_work_bytes=None):
    """DSYSV/no-predictions bounded route: reserve envelope PLUS the full cap."""
    envelope, work = fit_envelope(npoint, channels, parameters, frequencies, block_rows=block_rows)
    native = native_row_plan(npoint, parameters, frequencies, block_rows=block_rows,
                             maximum_work_bytes=maximum_work_bytes)
    return dict(python_fit_bytes=envelope, complete_refinement_bytes=envelope+native['native_work_bytes'],
                complete_refinement_work=work, **native)
