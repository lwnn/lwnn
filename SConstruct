import sys
sys.path.append('tools')
from building import *

PrepareEnv()

objs = scons('SConscript')
gtest_objs = scons('gtest/SConscript') + objs
Building('lwnn_gtest', gtest_objs)


