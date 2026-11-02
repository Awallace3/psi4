"""Finite descriptor-only scratch snapshots; never a reservation/RSS promise."""
import re


def _count(value):
    if type(value) is not str or re.fullmatch(r'\d+\*?', value) is None:
        raise ValueError('numeric unscaled Lustre quota output required')
    return int(value.rstrip('*'))


def parse_quota(text, filesystem):
    """Parse the aggregate unscaled `lfs quota` row, including wrapped paths."""
    if type(text) is not str or type(filesystem) is not str:
        raise ValueError('explicit quota text and filesystem required')
    rows, waiting = [], False
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == filesystem:
            if len(fields) == 1:
                waiting = True
                continue
            rows.append(fields[1:])
            waiting = False
        elif waiting:
            rows.append(fields)
            waiting = False
    if waiting or len(rows) != 1 or len(rows[0]) != 8:
        raise ValueError('one complete aggregate Lustre quota row required')
    used, soft, hard = map(_count, rows[0][:3])
    files, isoft, ihard = map(_count, rows[0][4:7])
    block_limits = [value for value in (soft, hard) if value]
    inode_limits = [value for value in (isoft, ihard) if value]
    limit = min(block_limits)*1024 if block_limits else None
    inode_limit = min(inode_limits) if inode_limits else None
    return dict(used_bytes=used*1024, limit_bytes=limit,
        remaining_bytes=None if limit is None else max(0, limit-used*1024),
        used_files=files, inode_limit=inode_limit,
        remaining_files=None if inode_limit is None else max(0, inode_limit-files),
        qualification='Quota snapshot; zero limits mean no ceiling reported, not reserved space.')


def admit_scratch(*, free_bytes, free_files, user_quota, group_quota, required_bytes, required_files):
    for name, value in (('free_bytes', free_bytes), ('free_files', free_files), ('required_bytes', required_bytes),
                        ('required_files', required_files)):
        if type(value) is not int or value <= 0:
            raise ValueError('explicit positive scratch snapshot budgets required: '+name)
    if free_bytes < required_bytes:
        raise ValueError('insufficient descriptor scratch filesystem free space')
    if free_files < required_files:
        raise ValueError('insufficient descriptor scratch filesystem free inodes')
    for owner, quota in (('user', user_quota), ('group', group_quota)):
        for key, needed in (('remaining_bytes', required_bytes), ('remaining_files', required_files)):
            value = quota[key]
            if value is not None and (type(value) is not int or value < needed):
                raise ValueError('insufficient descriptor scratch '+owner+' '+key)
    return dict(required_bytes=required_bytes, required_files=required_files,
        filesystem_free_bytes=free_bytes, filesystem_free_files=free_files,
        user_quota=user_quota, group_quota=group_quota,
        qualification='Descriptor-only free-space/quota snapshot, not a reservation, '
                      'SCF scratch bound or guarantee against concurrent usage.')
