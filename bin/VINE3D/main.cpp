#include <QtNetwork/QNetworkProxy>
#include <QMessageBox>
#include <QtGui>
#include <QApplication>

#include "commonConstants.h"
#include "vine3DProject.h"
#include "mainWindow.h"


Vine3DProject myProject;


int main(int argc, char *argv[])
{
    // set modality (default: GUI)
    if (argc > 1)
    {
        QString arg1 = QString::fromStdString(argv[1]);
        if (arg1.toUpper() == "CONSOLE")
        {
            myProject.modality = MODE_CONSOLE;
        }
        else
        {
            myProject.modality = MODE_BATCH;
        }
    }

    QApplication myApp(argc, argv);

    QNetworkProxyFactory::setUseSystemConfiguration(true);

    if (! myProject.start(myApp.applicationDirPath()))
        return -1;

    if (! myProject.loadParameters("parameters.ini"))
        return -1;

    if (myProject.modality == MODE_GUI)
    {
        QApplication::setOverrideCursor(Qt::ArrowCursor);
        MainWindow w;
        w.show();
        return myApp.exec();
    }
    else if (myProject.modality == MODE_CONSOLE)
    {
        return myProject.vine3dShell();
    }
    else if (myProject.modality == MODE_BATCH)
    {
        return myProject.vine3dBatch(argv[1]);
    }
}
