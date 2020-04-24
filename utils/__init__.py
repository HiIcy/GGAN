import os
import shutil


def mkdirs(dir,need_remove=False):
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        if need_remove:
            shutil.rmtree(dir)
            os.mkdir(dir)