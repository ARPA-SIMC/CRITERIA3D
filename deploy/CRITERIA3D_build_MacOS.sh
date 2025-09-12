#!/bin/bash

# specify your Qt directory
QT_DIR=/Users/xxxxxxx/Qt/5.x.x/clang_64/bin
QMAKE=$QT_DIR/qmake
QDEPLOY=$QT_DIR/macdeployqt

# build mapGraphics
cd ../mapGraphics
$QMAKE MapGraphics.pro -spec macx-clang CONFIG+=release CONFIG+=force_debug_info CONFIG+=x86_64 CONFIG+=qtquickcompiler
make -f Makefile clean
make -f Makefile all

export LD_LIBRARY_PATH=`pwd`/../mapGraphics/release/:$LD_LIBRARY_PATH

cd -

# build CRITERIA3D
cd ../bin/Makeall_CRITERIA3D
$QMAKE Makeall_CRITERIA3D.pro -spec macx-clang CONFIG+=release CONFIG+=force_debug_info CONFIG+=x86_64 CONFIG+=qtquickcompiler
make -f Makefile clean
make -f Makefile qmake_all
make 

cd -

# create bin directory and copy app
mkdir CRITERIA3D
mkdir CRITERIA3D/bin

cp -r ../bin/CRITERIA3D/CRITERIA3D.app CRITERIA3D/bin/CRITERIA3D.app

# deploy app
cd CRITERIA3D/bin
$DEPLOY CRITERIA3D.app

cd -

# copy doc and img
mkdir CRITERIA3D/DOC
mkdir CRITERIA3D/DOC/img
cp ../DOC/CRITERIA3D.pdf CRITERIA3D/DOC/CRITERIA3D.pdf
cp ../DOC/CRITERIA3D_user_manual.pdf CRITERIA3D/DOC/CRITERIA3D_user_manual.pdf
cp ../DOC/img/saveButton.png CRITERIA3D/DOC/img
cp ../DOC/img/updateButton.png CRITERIA3D/DOC/img
cp ../DOC/img/textural_soil.png CRITERIA3D/DOC/img

# copy ALL data directory
mkdir CRITERIA3D/DATA
cp -R ../DATA/* CRITERIA3D/DATA/

