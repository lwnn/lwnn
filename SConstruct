import sys
sys.path.append('tools')
from building import *

PrepareEnv()

objs = scons('SConscript')
gtenv = ForkEnv()
Export('gtenv')
gtest_objs = scons('gtest/SConscript') + objs
gtenv.Program('lwnn_gtest', gtest_objs)

if(not GetOption('android')):
    pyenv = ForkEnv()
    Export('pyenv')
    py_objs = scons('nn/python/SConscript') + objs
    target = 'tools/liblwnn.pyd'
    if(not IsPlatformWindows()):
        target = 'tools/liblwnn.so'
    pyenv.SharedLibrary(target, py_objs)
