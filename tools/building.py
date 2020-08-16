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
            proc = subprocess.Popen(cmdline, env=_e, shell=True)
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
    (output, _) = p.communicate()
    p_status = p.wait()
    return p_status, output.decode('utf-8')

def RunCommand(cmd, e=True):
    if(GetOption('verbose')):
        print(' >> RunCommand "%s"'%(cmd))
    if(os.name == 'nt'):
        cmd = cmd.replace('&&', '&')
    ret = os.system(cmd)
    if(0 != ret and e):
        raise Exception('FAIL of RunCommand "%s" = %s'%(cmd, ret))
    return ret

def AdbPush(src, tgt):
    tgt = tgt.replace(os.sep, '/')
    _, md5src= RunSysCmd('md5sum %s'%(src))
    md5src = md5src.split()[0]
    err, md5tgt= RunSysCmd('adb shell md5sum %s'%(tgt))
    if(0 == err):
        md5tgt = md5src.split()[0]
    if(md5tgt == md5src):
        print('skip push %s'%(src))
    else:
        cmd = 'adb push %s %s'%(src, tgt)
        cmd = cmd.replace(os.sep, '/')
        RunCommand(cmd)

def MKObject(src, tgt, cmd, rm=True, e=True):
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
        RunCommand(cmd, e)

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

    asenv=Environment(TOOLS=['as','ar','gcc','g++','gnulink'])
    os.environ['LWNN_ROOT'] = LWNN_ROOT
    asenv['LWNN_ROOT'] = LWNN_ROOT
    asenv['PACKAGES'] = []

    PrepareBuilding(asenv)
    if(Env == None):
        Env = asenv
        Export('asenv')
    return asenv

def ForkEnv(father=None, attr={}):
    if(father is None):
        father = Env
    child = Environment()
    for key,v in father.items():
        if(key == 'PACKAGES'):
            continue
        if(type(v) is list):
            child[key] = list(v)
        elif(type(v) is str):
            child[key] = str(v)
        elif(type(v) is dict):
            child[key] = dict(v)
        elif(type(v) is SCons.Util.CLVar):
            child[key] = SCons.Util.CLVar(v)
        else:
            child[key] = v
    for key,v in attr.items():
        child[key] = v
    return child

def PrepareBuilding(env):
    env['mingw64'] = False
    if(IsPlatformWindows()):
        mpath = os.path.abspath(os.getenv('MSYS2').replace('"',''))
        err,txt = RunSysCmd('which gcc')
        if(0 != err):
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
    env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME']=True
    env.Append(CCFLAGS=['-fPIC'])
    # add comstr option
    AddOption('--verbose',
            dest='verbose',
            action='store_true',
            default=False,
            help='print verbose information during build')
    AddOption('--android',
            dest='android',
            action='store_true',
            default=False,
            help='build android target')
    if(not GetOption('verbose')):
    # override the default verbose command string
        env.Replace(
          ARCOMSTR = 'AR $TARGET',
          ASCOMSTR = 'AS $SOURCE',
          ASPPCOMSTR = 'AS $SOURCE',
          CCCOMSTR = 'CC $SOURCE',
          CXXCOMSTR = 'CXX $SOURCE',
          LINKCOMSTR = 'LINK $TARGET',
          SHCCCOMSTR = 'SHCC $SOURCE',
          SHCXXCOMSTR = 'SHCXX $SOURCE',
          SHLINKCOMSTR = 'SHLINK $TARGET'
        )
    if(GetOption('android')):
        SelectCompilerAndroid(env)
    env.Append(CCFLAGS=['-g', '-O0'])

def SelectCompilerAndroid(env):
    HOME = os.getenv('HOME')
    NDK = os.path.join(HOME, 'AppData/Local/Android/Sdk/ndk-bundle')
    if(not os.path.exists(NDK)):
        NDK = os.getenv('ANDROID_NDK')
    if(not os.path.exists(NDK)):
        print('==> Please set environment ANDROID_NDK\n\tset ANDROID_NDK=/path/to/android-ndk')
        exit()
    if(IsPlatformWindows()):
        host = 'windows'
        NDK = NDK.replace(os.sep, '/')
    else:
        host = 'linux'
    env['ANDROID_NDK'] = NDK
    GCC = NDK + '/toolchains/llvm/prebuilt/%s-x86_64'%(host)
    env['CC']   = GCC + '/bin/aarch64-linux-android28-clang'
    env['AS']   = GCC + '/bin/aarch64-linux-android28-clang'
    env['CXX']  = GCC + '/bin/aarch64-linux-android28-clang++'
    env['LINK'] = GCC + '/bin/aarch64-linux-android28-clang++'

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

def MKSymlink(src,dst):
    asrc = os.path.abspath(src)
    adst = os.path.abspath(dst)

    if(not os.path.exists(dst)):
        if(IsPlatformWindows()):
            RunSysCmd('del %s'%(adst))
            if((sys.platform == 'msys') and
               (os.getenv('MSYS') == 'winsymlinks:nativestrict')):
                RunCommand('ln -fs %s %s'%(asrc,adst))
            elif(os.path.isdir(asrc)):
                RunCommand('mklink /D %s %s'%(adst,asrc))
            else:
                RunCommand('mklink %s %s'%(adst,asrc))
        else:
            RunSysCmd('rm -f %s'%(adst))
            os.symlink(asrc,adst)

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
    download = '%s/nn/third_party'%(Env['LWNN_ROOT'])
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

def SrcRemove(src, remove):
    if not src:
        return

    for item in src:
        if type(item) == type('str'):
            if(os.path.basename(item) in remove):
                src.remove(str(item))
        else:
            if(type(item) == list):
                for itt in item:
                    if(os.path.basename(itt.rstr()) in remove):
                        item.remove(itt)
                continue
            if(os.path.basename(item.rstr()) in remove):
                src.remove(item)

def AddPythonDev(env):
    pyp = sys.executable
    if(IsPlatformWindows()):
        pyp = pyp.replace(os.sep, '/')[:-10]
        pylib = 'python'+sys.version[0]+sys.version[2]
        if(pylib in env.get('LIBS',[])): return
        pf = '%s/libs/lib%s.a'%(pyp, pylib)
        if(not os.path.exists(pf)):
            RunCommand('cp {0}/libs/{1}.lib {0}/libs/lib{1}.a'.format(pyp, pylib))
        env.Append(CPPDEFINES=['_hypot=hypot'])
        env.Append(CPPPATH=['%s/include'%(pyp)])
        env.Append(LIBPATH=['%s/libs'%(pyp)])
        istr = 'set'
    else:
        pyp = os.sep.join(pyp.split(os.sep)[:-2])
        if(sys.version[0:3] == '2.7'):
            pylib = 'python'+sys.version[0:3]
        else:
            pylib = 'python'+sys.version[0:3]+'m'
        if(pylib in env.get('LIBS',[])): return
        env.Append(CPPPATH=['%s/include/%s'%(pyp,pylib)])
        if(pyp == '/usr'):
            env.Append(LIBPATH=['%s/lib/x86_64-linux-gnu'%(pyp)])
            env.Append(CPPPATH=['%s/local/include/%s'%(pyp,pylib[:9])])
        else:
            env.Append(LIBPATH=['%s/lib'%(pyp)])
        istr = 'export'
    #print('%s PYTHONHOME=%s if see error " Py_Initialize: unable to load the file system codec"'%(istr, pyp))
    env.Append(LIBS=[pylib, 'stdc++', 'dl', 'm'])

def scons(script):
    base = 'build'
    if(GetOption('android')):
        base += '/android'
    else:
        base += '/'+os.name
    bdir = '%s/%s'%(base,os.path.dirname(script))
    return SConscript(script, variant_dir=bdir, duplicate=0)

