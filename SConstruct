import os

env=Environment(TOOLS=['as','gcc','g++','gnulink'])
Export('env')
if(os.name=='nt'):
    env.AppendENVPath('PATH', os.getenv('PATH'))
    cuda = os.getenv('CUDA_PATH')
    env.Append(CPPPATH=['%s/include'%(cuda)])

env.Append(LIBS=['gtest','gtest_main','pthread'])
objs = SConscript('SConscript', variant_dir='build', duplicate=0)

if(os.name=='nt'):
    objs += ['%s/lib/x64/OpenCL.lib'%(cuda)]

env.Program('lwnn_gtest', objs)
