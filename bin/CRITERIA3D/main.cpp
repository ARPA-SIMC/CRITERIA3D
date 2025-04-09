#include <QApplication>
#include <QtNetwork/QNetworkProxy>
#include <QMessageBox>

#include "mainwindow.h"
#include "criteria3DProject.h"


Crit3DProject myProject;


bool setProxy(QString hostName, unsigned short port)
{
    QNetworkProxy myProxy;

    myProxy.setType(QNetworkProxy::HttpProxy);
    myProxy.setHostName(hostName);
    myProxy.setPort(port);

    try {
       QNetworkProxy::setApplicationProxy(myProxy);
    }
    catch (...) {
        QMessageBox::information(nullptr, "Error in proxy configuration!", "");
        return false;
    }

    return true;
}


int main(int argc, char *argv[])
{
    QApplication myApp(argc, argv);

    // set modality (default: GUI)
    myProject.modality = MODE_GUI;

    if (argc > 1)
    {
        QString arg1 = QString::fromStdString(argv[1]);
        if (arg1.toUpper() == "CONSOLE" || arg1.toUpper() == "SHELL")
        {
            myProject.modality = MODE_CONSOLE;
        }
        else
        {
            myProject.modality = MODE_BATCH;
        }
    }

    QNetworkProxyFactory::setUseSystemConfiguration(true);

    if (! myProject.start(myApp.applicationDirPath()))
    {
        myProject.logError();
        return -1;
    }

    if (! myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini"))
    {
        myProject.logWarning();
    }

    QApplication::setOverrideCursor(Qt::ArrowCursor);

    MainWindow mainWindow;
    mainWindow.show();

    return myApp.exec();
}

