"""Remove only this payload's fresh temporary root, after descendant reaping."""
import os
from pathlib import Path
import shutil
import stat


class OwnedScratch:
    def __init__(self,path,created,output):
        self.path=Path(path)
        if (not self.path.is_absolute() or str(self.path)!=created or self.path.is_symlink()
                or self.path.resolve()!=self.path
                or Path(output).resolve().is_relative_to(self.path)):
            raise RuntimeError("Scratch must be this payload's own fresh external directory")
        value=self.path.stat()
        if (not stat.S_ISDIR(value.st_mode) or value.st_uid!=os.getuid()
                or stat.S_IMODE(value.st_mode)!=0o700
                or not shutil.rmtree.avoids_symlink_attacks):
            raise RuntimeError("Owned private scratch/safe recursive deletion required")
        self.identity=(value.st_dev,value.st_ino)

    def cleanup(self,live_groups):
        if live_groups:
            raise RuntimeError("Cannot clean scratch before all owned descendants are reaped")
        try:value=self.path.lstat()
        except FileNotFoundError:return {"complete":True,"already_absent":True}
        if (not stat.S_ISDIR(value.st_mode) or value.st_uid!=os.getuid()
                or (value.st_dev,value.st_ino)!=self.identity):
            raise RuntimeError("Owned scratch root changed; refusing deletion")
        shutil.rmtree(self.path)
        return {"complete":True,"removed_owned_root":str(self.path)}
