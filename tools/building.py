import os
import glob
import sys
import shutil
import string
import re
from SCons.Script import *

Env = None

class Win32Spawn:
    def spawn(self, sh, escape, cmd, args, env):
        # deal with the cmd build-in commands which cannot be used in
        # subprocess.Popen
        if cmd == 'del':
            for f in args[1:]:
                try:
                    os.remove(f)
                except Exception as e:
                    print('Error removing file: %s'%(e))
                    return -1
            return 0

        import subprocess

        newargs = ' '.join(args[1:])
        cmdline = cmd + " " + newargs

        # Make sure the env is constructed by strings
        _e = dict([(k, str(v)) for k, v in env.items()])

        # Windows(tm) CreateProcess does not use the env passed to it to find
        # the executables. So we have to modify our own PATH to make Popen
        # work.
        old_path = os.environ['PATH']
        os.environ['PATH'] = _e['PATH']

        try:
            _e['PATH'] = env['EXTRAPATH']+';'+_e['PATH']
        except KeyError:
            pass

        try:
            proc = subprocess.Popen(cmdline, env=_e, shell=False)
        except Exception as e:
            print('Error in calling:\n%s'%(cmdline))
            print('Exception: %s: %s'%(e, os.strerror(e.errno)))
            return e.errno
        finally:
            os.environ['PATH'] = old_path

        return proc.wait()

def GetCurrentDir():
    conscript = File('SConscript')
    fn = conscript.rfile()
    path = os.path.dirname(fn.abspath)
    return os.path.abspath(path)

def IsPlatformWindows():
    bYes = False
    if(os.name == 'nt'):
        bYes = True
    if(sys.platform == 'msys'):
        bYes = True
    return bYes

def RunSysCmd(cmd):
    import subprocess
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    return err, output.decode('utf-8')

def AppendPythonPath(lp):
    try:
        pypath = os.environ['PYTHONPATH']
    except KeyError:
        pypath = ''
    sep = ':'
    if(IsPlatformWindows()):
        sep = ';'
    for l in lp:
        pypath += sep+os.path.abspath(l)
        sys.path.append(os.path.abspath(l))
    os.environ['PYTHONPATH'] = pypath

def PrepareEnv():
    global Env
    LWNN_ROOT = os.getenv('LWNN_ROOT')
    if((LWNN_ROOT==None) or (not os.path.exists(LWNN_ROOT))):
        # loop to search the LWNN_ROOT
        p = os.curdir
        while(True):
            if(os.path.isdir('%s/nn'%(p)) and 
               os.path.isdir('%s/gtest'%(p)) and
               os.path.isfile('%s/README.md'%(p))): break
            p=os.path.abspath('%s/..'%(p))
        LWNN_ROOT=p

    AppendPythonPath(['%s/tools'%(LWNN_ROOT)])

    asenv=Environment(TOOLS=['as','gcc','g++','gnulink'])
    os.environ['LWNN_ROOT'] = LWNN_ROOT
    asenv['LWNN_ROOT'] = LWNN_ROOT
    asenv['PACKAGES'] = []

    PrepareBuilding(asenv)
    if(Env == None):
        Env = asenv
        Export('asenv')

    return asenv

def PrepareBuilding(env):
    env['mingw64'] = False
    if(IsPlatformWindows()):
        mpath = os.path.abspath(os.getenv('MSYS2').replace('"',''))
        err,txt = RunSysCmd('which gcc')
        if(None != err):
            print('ERROR: not msys2 enviroment!')
            exit(-1)
        gcc = os.path.abspath(mpath+txt).strip()
        gccpath = os.path.dirname(gcc)
        if('mingw64' in gcc):
            env['mingw64'] = True
        env['CC'] = gcc
        env['LINK'] = gcc
        env['EXTRAPATH'] = '{0};{1}/usr/bin'.format(gccpath,mpath)
    env['python3'] = 'python3'
    if(IsPlatformWindows()):
        env['python3'] = 'python'
        env.AppendENVPath('PATH', os.getenv('PATH'))
        win32_spawn = Win32Spawn()
        env['SPAWN'] = win32_spawn.spawn
    env['CXX'] = env['CC']
    # add comstr option
    AddOption('--verbose',
            dest='verbose',
            action='store_true',
            default=False,
            help='print verbose information during build')
    AddOption('--gtest',
            dest='gtest',
            action='store_true',
            default=False,
            help='build LWNN gtest')
    if(not GetOption('verbose')):
    # override the default verbose command string
        env.Replace(
          ARCOMSTR = 'AR $SOURCE',
          ASCOMSTR = 'AS $SOURCE',
          ASPPCOMSTR = 'AS $SOURCE',
          CCCOMSTR = 'CC $SOURCE',
          CXXCOMSTR = 'CXX $SOURCE',
          LINKCOMSTR = 'LINK $TARGET'
        )

def scons(script):
    bdir = 'build/%s'%(os.path.dirname(script))
    return SConscript(script, variant_dir=bdir, duplicate=0)

def Building(target, objs, env=None):
    if(env is None):
        env = Env
    env.Program(target, objs)

