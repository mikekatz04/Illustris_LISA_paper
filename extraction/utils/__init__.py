

class SubProcess:

    def __init__(self, main_proc, **kwargs):
        print(self.__class__.__name__)
        self.ill_run = main_proc.ill_run
        self.dir_input = main_proc.dir_input
        self.dir_output = main_proc.dir_output

        self.first_snap_with_bhs = main_proc.first_snap_with_bhs
        self.skip_snaps = main_proc.skip_snaps
        self.max_snap = main_proc.max_snap
        self.base_url = "http://www.illustris-project.org/api/Illustris-%i/" % self.ill_run

        return
