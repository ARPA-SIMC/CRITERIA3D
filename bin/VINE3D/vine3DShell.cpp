#include "shell.h"
#include "vine3DProject.h"


QStringList getVine3DCommandList()
{
    QStringList cmdList = getSharedCommandList();

    cmdList.append("List    | ListCommands");
    cmdList.append("Proj    | OpenProject");
    cmdList.append("Run     | RunModels");

    return cmdList;
}


void Vine3DProject::cmdVine3dList()
{
    QStringList list = getVine3DCommandList();

    logInfo("Available VINE3D console commands:");
    logInfo("(short  | long version)");
    for (int i = 0; i < list.size(); i++)
    {
        logInfo(list[i]);
    }
}


bool Vine3DProject::cmdOpenVine3DProject(QStringList argumentList)
{
    if (argumentList.size() < 2)
    {
        logError("Missing project name");
        return false;
    }

    QString projectName = getCompleteFileName(argumentList.at(1), "PROJECT/");

    if (! loadVine3DProject(projectName))
    {
        logError();
        return false;
    }

    return true;
}


bool Vine3DProject::cmdRunModels(QStringList argumentList)
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

            logInfo(stringUsage);
            return true;
        }
    }

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

    QDate today = QDate::currentDate();
    QDate firstDay = today.addDays(-nrDays);
    QDate lastDay = today.addDays(nrDaysForecast);

    QDateTime firstDateTime;
    firstDateTime.setDate(firstDay);
    firstDateTime.setTime(QTime(1,0,0,0));

    QDateTime lastDateTime;
    lastDateTime.setDate(lastDay);
    lastDateTime.setTime(QTime(23,0,0,0));

    if (! runModels(firstDateTime, lastDateTime, true))
    {
        logError();
        return false;
    }

    return true;
}


bool Vine3DProject::vine3dShell()
{
#ifdef _WIN32
    openNewConsole();
#endif

    while (! requestedExit)
    {
        QString commandLine = getCommandLine("VINE3D");
        if (commandLine != "")
        {
            QStringList argumentList = getArgumentList(commandLine);
            executeCommand(argumentList);
        }
    }

    return true;
}


bool Vine3DProject::vine3dBatch(QString scriptFileName)
{
#ifdef _WIN32
    attachOutputToConsole();
#endif

    logInfo("\nVINE3D v1.0");
    logInfo("Execute script: " + scriptFileName);

    if (scriptFileName == "")
    {
        logError("No script file provided");
        return false;
    }

    if (! QFile(scriptFileName).exists())
    {
        logError("Script file not found: " + scriptFileName);
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

            if (! executeCommand(commandLine))
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


bool Vine3DProject::executeVine3DCommand(QStringList argumentList, bool *isCommandFound)
{
    *isCommandFound = false;
    if (argumentList.size() == 0) return false;

    QString command = argumentList.at(0).toUpper();

    if (command == "PROJ" || command == "OPENPROJECT")
    {
        *isCommandFound = true;
        return cmdOpenVine3DProject(argumentList);
    }
    else if (command == "RUN" || command == "RUNMODELS")
    {
        *isCommandFound = true;
        return cmdRunModels(argumentList);
    }
    if (command == "LIST" || command == "LISTCOMMANDS")
    {
        *isCommandFound = true;
        cmdVine3dList();
        return true;
    }
    else
    {
        // TODO:
        // other vine3d commands
    }

    return false;
}


bool Vine3DProject::executeCommand(QStringList argumentList)
{
    if (argumentList.size() == 0) return false;
    bool isCommandFound, isExecuted;

    logInfo(getTimeStamp(argumentList));

    isExecuted = executeSharedCommand(this, argumentList, &isCommandFound);
    if (isCommandFound)
        return isExecuted;

    isExecuted = executeVine3DCommand(argumentList, &isCommandFound);
    if (isCommandFound)
    {
        return isExecuted;
    }
    else
    {
        logError("This is not a valid VINE3D command.");
        return false;
    }
}

