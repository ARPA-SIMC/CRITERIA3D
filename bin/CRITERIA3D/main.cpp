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

    if (! myProject.start(myApp.applicationDirPath()))
    {
        myProject.logError();
        return -1;
    }

    if (! myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini"))
    {
        myProject.logWarning();
    }

    QNetworkProxyFactory::setUseSystemConfiguration(true);

    QApplication::setOverrideCursor(Qt::ArrowCursor);

    MainWindow mainWindow;
    mainWindow.show();

    return myApp.exec();
}

