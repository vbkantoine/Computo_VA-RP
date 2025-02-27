# init bayes_frag module
# Copyright (C) 2022 Antoine Van Biesbroeck
#
#
# Licence

import os
import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])) # get script's path
# os.chdir(directory)
import sys
sys.path.append(directory)
sys.path.append(os.path.join(directory, "../"))
sys.path.append(os.getcwd())

# global __directory__
# __directory__ = directory