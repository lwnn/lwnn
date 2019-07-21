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

def RunCommand(cmd, e=True):
    if(GetOption('verbose')):
        print(' >> RunCommand "%s"'%(cmd))
    if(os.name == 'nt'):
        cmd = cmd.replace('&&', '&')
    ret = os.system(cmd)
    if(0 != ret and e):
        raise Exception('FAIL of RunCommand "%s" = %s'%(cmd, ret))
    return ret

def MKObject(src, tgt, cmd, rm=True):
    if(GetOption('clean') and rm):
        RMFile(tgt)
        return
    mtime = 0
    for s in src:
        s = str(s)
        if(os.path.isfile(s)):
            tm = os.path.getmtime(s)
            if(tm > mtime):
                mtime = tm
    if(os.path.isfile(tgt)):
        mtime2 = os.path.getmtime(tgt)
    else:
        mtime2 = -1
    if(mtime2 < mtime):
        RunCommand(cmd)

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
        p = os.path.abspath(os.curdir)
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

def MKDir(p):
    ap = os.path.abspath(p)
    try:
        os.makedirs(ap)
    except:
        if(not os.path.exists(ap)):
            raise Exception('Fatal Error: can\'t create directory <%s>'%(ap))

def RMDir(p):
    if(os.path.exists(p)):
        shutil.rmtree(p)

def RMFile(p):
    if(os.path.exists(p)):
        print('removing %s'%(os.path.abspath(p)))
        os.remove(os.path.abspath(p))

def MKFile(p,c='',m='w'):
    f = open(p,m)
    f.write(c)
    f.close()

def Download(url, tgt=None):
    # curl is better than wget on msys2
    if(tgt == None):
        tgt = url.split('/')[-1]
    def IsProperType(f):
        tL = {'.zip':'Zip archive data', '.tar.gz':'gzip compressed data',
              '.tar.xz':'XZ compressed data','.tar.bz2':'bzip2 compressed data'}
        if(not os.path.exists(f)):
            return False
        if(0 == os.path.getsize(f)):
            return False
        for t,v in tL.items():
            if(f.endswith(t)):
                err,info = RunSysCmd('file %s'%(tgt))
                if(v not in info):
                    return False
                break
        return True
    if(not os.path.exists(tgt)):
        print('Downloading from %s to %s'%(url, tgt))
        ret = RunCommand('curl %s -o %s'%(url,tgt), False)
        if((ret != 0) or (not IsProperType(tgt))):
            tf = url.split('/')[-1]
            RMFile(tf)
            print('temporarily saving to %s'%(os.path.abspath(tf)))
            RunCommand('wget %s'%(url))
            RunCommand('mv -v %s %s'%(tf, tgt))

def Package(url, ** parameters):
    if(type(url) == dict):
        parameters = url
        url = url['url']
    download = GetCurrentDir()
    pkgBaseName = os.path.basename(url)
    if(pkgBaseName.endswith('.zip')):
        tgt = '%s/%s'%(download, pkgBaseName)
        Download(url, tgt)
        pkgName = pkgBaseName[:-4]
        pkg = '%s/%s'%(download, pkgName)
        MKDir(pkg)
        flag = '%s/.unzip.done'%(pkg)
        if(not os.path.exists(flag)):
            try:
                RunCommand('cd %s && unzip ../%s'%(pkg, pkgBaseName))
            except Exception as e:
                print('WARNING:',e)
            MKFile(flag,'url')
    elif(pkgBaseName.endswith('.rar')):
        tgt = '%s/%s'%(download, pkgBaseName)
        Download(url, tgt)
        pkgName = pkgBaseName[:-4]
        pkg = '%s/%s'%(download, pkgName)
        MKDir(pkg)
        flag = '%s/.unrar.done'%(pkg)
        if(not os.path.exists(flag)):
            try:
                RunCommand('cd %s && unrar x ../%s'%(pkg, pkgBaseName))
            except Exception as e:
                print('WARNING:',e)
            MKFile(flag,'url')
    elif(pkgBaseName.endswith('.tar.gz') or pkgBaseName.endswith('.tar.xz')):
        tgt = '%s/%s'%(download, pkgBaseName)
        Download(url, tgt)
        pkgName = pkgBaseName[:-7]
        pkg = '%s/%s'%(download, pkgName)
        MKDir(pkg)
        flag = '%s/.unzip.done'%(pkg)
        if(not os.path.exists(flag)):
            RunCommand('cd %s && tar xf ../%s'%(pkg, pkgBaseName))
            MKFile(flag,'url')
    elif(pkgBaseName.endswith('.tar.bz2')):
        tgt = '%s/%s'%(download, pkgBaseName)
        Download(url, tgt)
        pkgName = pkgBaseName[:-8]
        pkg = '%s/%s'%(download, pkgName)
        MKDir(pkg)
        flag = '%s/.unzip.done'%(pkg)
        if(not os.path.exists(flag)):
            RunCommand('cd %s && tar xf ../%s'%(pkg, pkgBaseName))
            MKFile(flag,'url')
    elif(pkgBaseName.endswith('.git')):
        pkgName = pkgBaseName[:-4]
        pkg = '%s/%s'%(download, pkgName)
        if(not os.path.exists(pkg)):
            RunCommand('cd %s && git clone %s'%(download, url))
        if('version' in parameters):
            flag = '%s/.version.done'%(pkg)
            if(not os.path.exists(flag)):
                ver = parameters['version']
                RunCommand('cd %s && git checkout %s'%(pkg, ver))
                MKFile(flag,ver)
                # remove all cmd Done flags
                for cmdF in Glob('%s/.*.cmd.done'%(pkg)):
                    RMFile(str(cmdF))
    else:
        pkg = '%s/%s'%(download, url)
        if(not os.path.isdir(pkg)):
            print('ERROR: require %s but now it is missing!'
                  ' It maybe downloaded later, so please try build again.'%(url))
    # cmd is generally a series of 'sed' operatiron to do some simple modifications
    if('cmd' in parameters):
        flag = '%s/.cmd.done'%(pkg)
        cmd = 'cd %s && '%(pkg)
        cmd += parameters['cmd']
        if(not os.path.exists(flag)):
            RunCommand(cmd)
            MKFile(flag,cmd)
    if('pyfnc' in parameters):
        flag = '%s/.pyfnc.done'%(pkg)
        if(not os.path.exists(flag)):
            parameters['pyfnc'](pkg)
            MKFile(flag)
    return pkg

def scons(script):
    bdir = 'build/%s'%(os.path.dirname(script))
    return SConscript(script, variant_dir=bdir, duplicate=0)

def Building(target, objs, env=None):
    if(env is None):
        env = Env
    env.Program(target, objs)

