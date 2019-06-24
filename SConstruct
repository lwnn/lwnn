import os

env=Environment(TOOLS=['as','gcc','g++','gnulink'])
if(os.name=='nt'):
    env.AppendENVPath('PATH', os.getenv('PATH'))

objs = SConscript('SConscript', variant_dir='build', duplicate=0)

env.Program('nncl', objs)
