    
##	Build CRITERIA3D
 
 
###	Dependencies

	Qt libraries: Qt 5.x or following is needed (download also QtCharts).
 
 
###	Notes for Windows
 
	Compiler:
	MSVC = Microsoft Visual C++ compiler
	MinGW = MinGW C++ compiler
 
	build:
	- open Qt shell (MSVC or MinGW version)
	- move to deploy directory (cd [local path]\CRITERIA3D\deploy)
	- execute CRITERIA3D_Build_Win_XXXX.bat 
	
	Only for MSVC:
	before build execution call vcvarsall.bat to complete environment setup (with x64 option for 64-bit compilers)
	example: 
	> cd C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build
	> vcvarsall.bat x64
 

###	Notes for Linux

	Compiler:gcc v.6 or later
	Qt version 5.x or later
        
	build:
	- execute CRITERIA3D_build_Linux.sh
