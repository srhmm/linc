import os

class Out:
    """For debug and logging. """
    def __init__(self, file_nm, vb = True, tofile = True):
        self.file_nm = file_nm
        self.vb = vb
        self.tofile = tofile
        if self.tofile:
            try:
                os.mkdir("logs/")
                pass
            except(FileExistsError):
                pass
            self.file = open("logs/"+str(self.file_nm), "a+")

    def printto(self, *args):
        if self.vb:
            if self.tofile:
                print(*args, file=self.file)
            else:
                print(*args)
        return

    def close(self):
        if self.tofile:
            self.file.close()
