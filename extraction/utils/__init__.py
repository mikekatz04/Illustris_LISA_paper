"""
"""

import os
import sys


class SubProcess:

    def __init__(self, core, **kwargs):
        print(self.__class__.__name__)
        self.core = core
        self.ill_run = core.ill_run
        self.dir_input = core.dir_input
        self.dir_output = core.dir_output

        self.first_snap_with_bhs = core.first_snap_with_bhs
        self.skip_snaps = core.skip_snaps
        self.max_snap = core.max_snap
        self.base_url = "http://www.illustris-project.org/api/Illustris-%i/" % self.ill_run

        return

    def _check_needed(self, *fnames):
        needed = False
        for fn in fnames:
            exists = os.path.exists(fn)
            if self.core.DEBUG:
                print("\tfile: '{}', exists: {}, ".format(fn, exists))
            if not exists:
                needed = True

        if not needed and self.core.RECREATE:
            print("\tRecreating existing files")
            needed = True

        if self.core.DEBUG:
            print("\tNeeded = ", needed)

        return needed
