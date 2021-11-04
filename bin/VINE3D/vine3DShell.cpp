#include <QFile>
#include <QTextStream>
#include "shell.h"
#include "project.h"
#include "vine3DShell.h"

QStringList getVine3DCommandList()
{
    QStringList cmdList = getSharedCommandList();

    cmdList.append("List    | ListCommands");
    cmdList.append("Proj    | OpenProject");
    cmdList.append("Run     | RunModels");

    return cmdList;
}

bool cmdList(Vine3DProject* myProject)
{
    QStringList list = getVine3DCommandList();

    myProject->logInfo("Available VINE3D console commands:");
    myProject->logInfo("(short  | long version)");
    for (int i = 0; i < list.size(); i++)
    {
        myProject->logInfo(list[i]);
    }

    return true;
}

bool Vine3DProject::executeVine3DCommand(QStringList argumentList, bool *isCommandFound)
{
    *isCommandFound = false;
    if (argumentList.size() == 0) return false;

    QString command = argumentList.at(0).toUpper();

    if (command == "PROJ" || command == "OPENPROJECT")
    {
        *isCommandFound = true;
        return cmdOpenVine3DProject(this, argumentList);
    }
    else if (command == "RUN" || command == "RUNMODELS")
    {
        *isCommandFound = true;
        return cmdRunModels(this, argumentList);
    }
    if (command == "LIST" || command == "LISTCOMMANDS")
    {
        *isCommandFound = true;
        return cmdList(this);
    }
    else
    {
        // TODO:
        // other vine3d commands
    }

    return false;
}

bool cmdOpenVine3DProject(Vine3DProject* myProject, QStringList argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing project name");
        return false;
    }

    QString projectName = myProject->getCompleteFileName(argumentList.at(1), "PROJECT/");

    if (! myProject->loadVine3DProject(projectName))
        return false;

    return true;
}

bool cmdRunModels(Vine3DProject* myProject, QStringList argumentList)
{
    if (argumentList.size() == 0) return false;

    if (argumentList.size() >= 2)
    {
        if (argumentList.at(1) == "?" || argumentList.at(1) == "-?")
        {
            QString stringUsage = "USAGE:";
            stringUsage += "\nrunModels [nrDaysPast] [nrDaysForecast]";
            stringUsage += "\nnrDaysPast: days from today when to start (default: 7)";
            stringUsage += "\nnrDaysForecast: days since today when to finish (default: 0)";

            myProject->logInfo(stringUsage);
            return true;
        }
    }

    QDate today, firstDay;
    int nrDays, nrDaysForecast;

    if (argumentList.size() >= 3)
    {
        nrDaysForecast = argumentList.at(2).toInt();
        nrDays = argumentList.at(1).toInt();
    }
    else if (argumentList.size() == 2)
    {
        nrDays = argumentList.at(1).toInt();
        nrDaysForecast = 0;
    }
    else
    {
        //default: 1 week
        nrDays = 7;
        nrDaysForecast = 0;
    }

    today = QDate::currentDate();
    QDateTime lastDateTime = QDateTime(today);
    lastDateTime = lastDateTime.addDays(nrDaysForecast);
    lastDateTime.setTime(QTime(23,0,0,0));

    firstDay = today.addDays(-nrDays);
    QDateTime firstDateTime = QDateTime(firstDay);
    firstDateTime.setTime(QTime(1,0,0,0));

    if (! myProject->runModels(firstDateTime, lastDateTime, true, true, myProject->idArea))
        return false;

    return true;
}


bool executeCommand(QStringList argumentList, Vine3DProject* myProject)
{
    if (argumentList.size() == 0) return false;
    bool isCommandFound, isExecuted;

    myProject->logInfo(getTimeStamp(argumentList));

    isExecuted = executeSharedCommand(myProject, argumentList, &isCommandFound);
    if (isCommandFound) return isExecuted;

    isExecuted = myProject->executeVine3DCommand(argumentList, &isCommandFound);
    if (isCommandFound) return isExecuted;

    myProject->logError("This is not a valid VINE3D command.");
    return false;
}


bool vine3dBatch(Vine3DProject *myProject, QString scriptFileName)
{
    #ifdef _WIN32
        attachOutputToConsole();
    #endif

    myProject->logInfo("\nVINE3D v1.0");
    myProject->logInfo("Execute script: " + scriptFileName);

    if (scriptFileName == "")
    {
        myProject->logError("No script file provided");
        return false;
    }

    if (! QFile(scriptFileName).exists())
    {
        myProject->logError("Script file not found: " + scriptFileName);
        return false;
    }

    QString line;
    QStringList commandLine;

    QFile inputFile(scriptFileName);
    if (inputFile.open(QIODevice::ReadOnly))
    {
       QTextStream in(&inputFile);
       while (!in.atEnd())
       {
          line = in.readLine();
          commandLine = line.split(" ");

          if (! executeCommand(commandLine, myProject))
          {
              inputFile.close();
              return false;
          }
       }
       inputFile.close();
    }

    #ifdef _WIN32
        // Send "enter" to release application from the console
        // This is a hack, but if not used the console doesn't know the application has
        // returned. The "enter" key only sent if the console window is in focus.
        if (isConsoleForeground()) sendEnterKey();
    #endif

    return true;
}


bool vine3dShell(Vine3DProject* myProject)
{
    #ifdef _WIN32
        openNewConsole();
    #endif

    while (! myProject->requestedExit)
    {
        QString commandLine = getCommandLine("VINE3D");
        if (commandLine != "")
        {
            QStringList argumentList = getArgumentList(commandLine);
            executeCommand(argumentList, myProject);
        }
    }

    return true;
}

