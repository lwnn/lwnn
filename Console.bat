@echo off

set ASPATH=%~dp0
set astmp=%ASPATH%
set ASDISK=%astmp:~1,2%
set MSYS2=C:\msys64
set INTELSW=C:\Program Files (x86)\IntelSWTools

%ASDISK%
cd %ASPATH%

echo %ASPATH%

if NOT EXIST "%ASPATH%\Console.bat" goto perror
if NOT EXIST %MSYS2%\usr\bin goto install_msys2
REM if NOT EXIST "%INTELSW%\openvino\bin\setupvars.bat" goto install_openvino

REM base env PATH
set PATH=C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0

set PATH=C:\Anaconda3;C:\Anaconda3\Scripts;%MSYS2%\mingw64\bin;%MSYS2%\usr\bin;%MSYS2%\mingw32\bin;%PATH%

if NOT EXIST "%ASPATH%\tools\download" mkdir %ASPATH%\tools\download

set ConEmu=%ASPATH%\tools\download\ConEmu\ConEmu64.exe

if EXIST %ConEmu% goto prepareEnv
cd %ASPATH%\tools\download
mkdir ConEmu
cd ConEmu
wget https://github.com/Maximus5/ConEmu/releases/download/v19.07.14/ConEmuPack.190714.7z
"C:\Program Files\7-Zip\7z.exe" x ConEmuPack.190714.7z
cd %ASPATH%

:prepareEnv
%INTELSW%\openvino\bin\setupvars.bat
set PYTHONPATH=%INTELSW%/openvino\python\python3.7;%PYTHONPATH%
set MSYS=winsymlinks:nativestrict
set PYTHONPATH=%ASPATH%/tools;%PYTHONPATH%
REM env.asc in format "tokens=value" to set some environment
if EXIST "%ASPATH%\env.asc" for /F "tokens=*" %%I in (%ASPATH%\env.asc) do set %%I
cd %ASPATH%
set ISMSYS2=YES

start %ConEmu% -title lwnn-gtest-misc ^
	-runlist -new_console:d:"%ASPATH%":t:lwnn ^
	^|^|^| -new_console:d:"%ASPATH%\gtest":t:gtest ^
	^|^|^| -new_console:d:"%ASPATH%":t:misc
exit 0

:install_msys2
set msys2="www.msys2.org"
echo Please visit %msys2% and install msys2 as c:\msys64
pause
exit -1

:install_openvino
echo Please visit https://software.intel.com/en-us/openvino-toolkit and install openvino as %INTELSW%
pause
exit -1

:perror
echo Please fix the var "ASDISK" and "ASPATH" to the right path!
pause

:exitPoint
