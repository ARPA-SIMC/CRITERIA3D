#include "mainGUI.h"
#include "mainWindow.h"
#include <QApplication>
#include <QMessageBox>

#include "criteria3DProject.h"


bool checkEnvironmentGUI(QString criteria3dHome)
{
    if (criteria3dHome.isEmpty())
    {
        QString warning = "Set CRITERIA3D_HOME in the environment variables:"
                          "\n$CRITERIA3D_HOME = path of CRITERIA3D directory";

        QMessageBox::critical(nullptr, "Missing environment", warning);
        return false;
    }

    if (! QDir(criteria3dHome).exists())
    {
        QString warning = criteria3dHome + "  doesn't exist!"
                                      "\nSet correct $CRITERIA3D_HOME in the environment variables:"
                                      "\n$CRITERIA3D_HOME = path of CRITERIA3D directory";

        QMessageBox::critical(nullptr, "Wrong environment!", warning);
        return false;
    }

    return true;
}


QString searchDefaultPath(QString startPath)
{
    QString myRoot = QDir::rootPath();
    QString path = startPath;

    // Installation on other volume (for example D:)
    QString myVolume = path.left(3);

    bool isFound = false;
    while (! isFound)
    {
        if (QDir(path + "/DATA").exists())
        {
            isFound = true;
            break;
        }
        if (QDir::cleanPath(path) == myRoot || QDir::cleanPath(path) == myVolume)
            break;

        path = QFileInfo(path).dir().absolutePath();
    }

    if (! isFound)
    {
        QMessageBox::critical(nullptr, "Wrong environment!", "DATA directory is missing");
        return "";
    }

    return QDir::cleanPath(path);
}


int mainGUI(int argc, char *argv[], QString criteria3dHome, Crit3DProject& myProject)
{
    QApplication myApp(argc, argv);
    QApplication::setOverrideCursor(Qt::ArrowCursor);

    // Windows without right to set environment
    if (QSysInfo::productType() == "windows" && criteria3dHome == "")
    {
        QString appPath = myApp.applicationDirPath();
        criteria3dHome = searchDefaultPath(appPath);
    }

    if (! checkEnvironmentGUI(criteria3dHome))
    {
        QString appPath = myApp.applicationDirPath();
        criteria3dHome = searchDefaultPath(appPath);
        if (! checkEnvironmentGUI(criteria3dHome))
            return -1;
    }

    if (! myProject.start(criteria3dHome))
        return -1;

    if (! myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini"))
        return -1;

    MainWindow w;
    w.show();

    return myApp.exec();
}
