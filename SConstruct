import sys
sys.path.append('tools')
from building import *

PrepareEnv()

objs = scons('SConscript')
gtest_objs = objs + scons('gtest/SConscript')
Building('lwnn_gtest', gtest_objs)


