#include <QCoreApplication>
#include <QtNetwork/QNetworkProxy>

#include "commonConstants.h"
#include "mainwindow.h"
#include "criteria3DProject.h"
#include "shell.h"
#include "mainGUI.h"


Crit3DProject myProject;


// check $CRITERIA3D_HOME
bool checkEnvironmentConsole(QString criteria3dHome)
{
    #ifdef _WIN32
        attachOutputToConsole();
    #endif

    if (criteria3dHome == "")
    {
        QString error = "\nSet CRITERIA3D_HOME in the environment variables:\n"
                        "$CRITERIA3D_HOME = path of CRITERIA3D directory\n";

        myProject.logError(error);
        return false;
    }

    if (! QDir(criteria3dHome).exists())
    {
        QString error = "\nWrong environment!\n"
                        "Set correct $CRITERIA3D_HOME variable:\n"
                        "$CRITERIA3D_HOME = path of CRITERIA3D directory\n";

        myProject.logError(error);
        return false;
    }

    return true;
}


int main(int argc, char *argv[])
{
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

    // read environment
    QProcessEnvironment myEnvironment = QProcessEnvironment::systemEnvironment();
    QString criteria3dHome = myEnvironment.value("CRITERIA3D_HOME");
    QString display = myEnvironment.value("DISPLAY");

    // only for headless Linux
    if (QSysInfo::productType() != "windows" && QSysInfo::productType() != "osx")
    {
        if (myProject.modality == MODE_GUI && display.isEmpty())
        {
            // server headless (computers without a local interface): switch modality
            myProject.modality = MODE_CONSOLE;
        }
    }

    if (myProject.modality == MODE_GUI)
    {
        // go to GUI starting point
        return mainGUI(argc, argv, criteria3dHome, myProject);
    }
    else
    {
        // SHELL
        QCoreApplication myApp(argc, argv);

        // only for Windows - without the right to set the environment
        if (QSysInfo::productType() == "windows" && criteria3dHome == "")
        {
            // search default praga home
            QString appPath = myApp.applicationDirPath();
            criteria3dHome = searchDefaultPath(appPath);
        }

        if (! checkEnvironmentConsole(criteria3dHome))
            return -1;

        if (! myProject.start(criteria3dHome))
        {
            myProject.logError();
            return -1;
        }

        if (! myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini"))
            return -1;


        // start modality
        if (myProject.modality == MODE_CONSOLE)
        {
            return myProject.criteria3DShell();
        }
        else if (myProject.modality == MODE_BATCH)
        {
            return myProject.criteria3DBatch(argv[1]);
        }


        return 0;
    }
}

