#include "shell.h"
#include "project.h"
#include "commonConstants.h"
#include <iostream>
#include <sstream>
#include <QString>
#include <QList>
#include <QFile>


#ifdef _WIN32
    #include "Windows.h"
    #pragma comment(lib, "User32.lib")
#endif



bool attachOutputToConsole()
{
    #ifdef _WIN32
        HANDLE consoleHandleOut, consoleHandleIn, consoleHandleError;

        if (AttachConsole(ATTACH_PARENT_PROCESS))
        {
            // Redirect unbuffered STDOUT to the console
            consoleHandleOut = GetStdHandle(STD_OUTPUT_HANDLE);
            if (consoleHandleOut != INVALID_HANDLE_VALUE)
            {
                freopen("CONOUT$", "w", stdout);
                setvbuf(stdout, nullptr, _IONBF, 0);
            }
            else
            {
                return false;
            }

            // Redirect STDIN to the console
            consoleHandleIn = GetStdHandle(STD_INPUT_HANDLE);
            if (consoleHandleIn != INVALID_HANDLE_VALUE)
            {
                freopen("CONIN$", "r", stdin);
                setvbuf(stdin, nullptr, _IONBF, 0);
            }
            else
            {
                return false;
            }

            // Redirect unbuffered STDERR to the console
            consoleHandleError = GetStdHandle(STD_ERROR_HANDLE);
            if (consoleHandleError != INVALID_HANDLE_VALUE)
            {
                freopen("CONOUT$", "w", stderr);
                setvbuf(stderr, nullptr, _IONBF, 0);
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            // Not a console application
            return false;
        }
    #endif

    return true;
}

bool isConsoleForeground()
{
    #ifdef _WIN32
        return (GetConsoleWindow() == GetForegroundWindow());
    #endif
}

void sendEnterKey(void)
{
    #ifdef _WIN32
        INPUT ip;
        // Set up a generic keyboard event.
        ip.type = INPUT_KEYBOARD;
        ip.ki.wScan = 0; // hardware scan code for key
        ip.ki.time = 0;
        ip.ki.dwExtraInfo = 0;

        // Send the "Enter" key
        ip.ki.wVk = 0x0D; // virtual-key code for the "Enter" key
        ip.ki.dwFlags = 0; // 0 for key press
        SendInput(1, &ip, sizeof(INPUT));

        // Release the "Enter" key
        ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
        SendInput(1, &ip, sizeof(INPUT));
    #endif
}

void openNewConsole()
{
    #ifdef _WIN32
        // detach from the current console window
        // if launched from a console window, that will still run waiting for the new console (below) to close
        // it is useful to detach from Qt Creator's <Application output> panel
        FreeConsole();

        // create a separate new console window
        AllocConsole();

        // attach the new console to this application's process
        AttachConsole(GetCurrentProcessId());

        // reopen the std I/O streams to redirect I/O to the new console
        freopen("CON", "w", stdout);
        freopen("CON", "w", stderr);
        freopen("CON", "r", stdin);
    #endif
}


QString getTimeStamp(QList<QString> argumentList)
{
    QString myString = ">> ";

    QDate myDate = QDateTime::currentDateTime().date();
    QTime myTime = QDateTime::currentDateTime().time();
    myString += QDateTime(myDate, myTime, Qt::UTC).toString("yyyy-MM-dd hh:mm:ss");
    myString += " >>";

    for (int i = 0; i < argumentList.size(); i++)
    {
        myString += " " + argumentList[i];
    }

    return myString;
}


QList<QString> getArgumentList(QString commandLine)
{
    std::string str;
    QList<QString> argumentList;

    std::istringstream stream(commandLine.toStdString());
    while (stream >> str)
    {
        argumentList.append(QString::fromStdString(str));
    }

    return argumentList;
}


QString getCommandLine(QString programName)
{
    std::string commandLine;

    std::cout << programName.toStdString() << ">";
    getline (std::cin, commandLine);

    return QString::fromStdString(commandLine);
}


QList<QString> getSharedCommandList()
{
    QList<QString> cmdList;

    cmdList.append("DEM         | LoadDEM");
    cmdList.append("Point       | LoadPoints");
    cmdList.append("Grid        | LoadGrid");
    cmdList.append("DailyCsv    | ExportDailyDataCsv");
    cmdList.append("Log         | SetLogFile");
    cmdList.append("Quit        | Exit");

    return cmdList;
}


int cmdExit(Project* myProject)
{
    myProject->requestedExit = true;

    // TODO: close project

    return PRAGA_OK;
}


int cmdLoadDEM(Project* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing DEM file name.");
        // TODO: USAGE
        return PRAGA_MISSING_FILE;
    }
    else
    {
        if (myProject->loadDEM(argumentList[1]))
        {
            return PRAGA_OK;
        }
        else
        {
            return PRAGA_ERROR;
        }
    }
}

int cmdOpenDbPoint(Project* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing db point name");
        return PRAGA_INVALID_COMMAND;
    }

    QString filename = argumentList.at(1);

    if (! myProject->loadMeteoPointsDB(filename))
    {
        myProject->logError();
        return ERROR_DBPOINT;
    }

    return PRAGA_OK;
}

int cmdLoadMeteoGrid(Project* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing Grid file name.");
        // TODO: USAGE
        return PRAGA_MISSING_FILE;
    }
    else
    {
        if (!myProject->loadMeteoGridDB(argumentList[1]))
        {
            return PRAGA_ERROR;
        }
        else
        {
            myProject->meteoGridDbHandler->meteoGrid()->createRasterGrid();
            return PRAGA_OK;
        }
    }
}


int cmdSetLogFile(Project* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing Log file name.");
        // TODO: USAGE
        return PRAGA_INVALID_COMMAND;
    }
    else
    {
        if (myProject->setLogFile(argumentList[1]))
        {
            return PRAGA_OK;
        }
        else
        {
            return PRAGA_ERROR;
        }
    }
}


int cmdExportDailyDataCsv(Project* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        QString usage = "Usage:\n"
                        "ExportDailyDataCsv [TPREC] -d1:firstDate [-d2:lastDate] [-t:GRID|POINTS] [-l:pointList] [-p:outputPath]\n"
                        "date format: YYYY-MM-DD \n"
                        "lastDate default: yesterday \n"
                        "TPREC: save only Tmin, Tmax, Tavg, Prec \n"
                        "type default: GRID \n"
                        "point list default: ALL \n"
                        "output path default: ProjectPath/OUTPUT/ \n";

        myProject->logInfo(usage);
        return PRAGA_OK;
    }

    QString typeStr = "GRID";
    QString pointListStr = "";
    QString outputPath = myProject->getProjectPath() + PATH_OUTPUT;
    QDate firstDate, lastDate;
    bool isTPrec = false;

    for (int i = 1; i < argumentList.size(); i++)
    {
        if (argumentList.at(i).left(5).toUpper() == "TPREC")
            isTPrec = true;

        if (argumentList.at(i).left(4) == "-d1:")
        {
            QString dateStr = argumentList[i].right(argumentList[i].length()-4);
            firstDate = QDate::fromString(dateStr, "yyyy-MM-dd");

            if (! firstDate.isValid())
            {
                myProject->logError("Wrong first date, required format is: YYYY-MM-DD");
                return PRAGA_OK;
            }
        }

        if (argumentList.at(i).left(4) == "-d2:")
        {
            QString dateStr = argumentList[i].right(argumentList[i].length()-4);
            lastDate = QDate::fromString(dateStr, "yyyy-MM-dd");

            if (! lastDate.isValid())
            {
                myProject->logError("Wrong last date, required format is: YYYY-MM-DD");
                return PRAGA_OK;
            }
        }

        if (argumentList.at(i).left(3) == "-t:")
        {
            typeStr = argumentList[i].right(argumentList[i].length()-3).toUpper();

            if (typeStr != "GRID" && typeStr != "POINTS")
            {
                myProject->logError("Wrong date, required format is: YYYY-MM-DD");
                return PRAGA_OK;
            }
        }

        if (argumentList.at(i).left(3) == "-l:")
        {
            pointListStr = argumentList[i].right(argumentList[i].length()-3);
            if (! QFile::exists(pointListStr))
            {
                myProject->logError("Wrong point list, the file does not exist: " + pointListStr);
                return PRAGA_OK;
            }
        }

        if (argumentList.at(i).left(3) == "-p:")
        {
            outputPath = argumentList[i].right(argumentList[i].length()-3);
            QDir dir(outputPath);
            if (! dir.exists())
            {
                if (! dir.mkpath(outputPath))
                {
                    myProject->logError("Wrong outputPath, the directory could not be created: " + outputPath);
                    return PRAGA_OK;
                }
            }
            outputPath = dir.absolutePath();
        }
    }

    // check first date (mandatory)
    if (! firstDate.isValid())
    {
        myProject->logError("Missing first date: use option -d1:firstDate");
        return PRAGA_OK;
    }
    // check last date (default: yesterday)
    if (! lastDate.isValid())
    {
        lastDate = QDateTime::currentDateTime().date().addDays(-1);
    }

    myProject->logInfo("... first date is: " + firstDate.toString());
    myProject->logInfo("... last date is: " + lastDate.toString());
    if (isTPrec)
    {
        myProject->logInfo("... output format is: Tmin, Tmax, Tavg, Prec");
    }
    if (pointListStr != "")
    {
        myProject->logInfo("... point list file is: " + pointListStr);
    }
    else
    {
        if (typeStr == "GRID")
            myProject->logInfo("... export ALL the cells of the meteoGrid");
        else
            myProject->logInfo("... export ALL meteo points");
    }
    myProject->logInfo("... output path is: " + outputPath);

    return PRAGA_OK;
}


int executeSharedCommand(Project* myProject, QList<QString> argumentList, bool* isCommandFound)
{
    *isCommandFound = false;
    if (argumentList.size() == 0) return PRAGA_INVALID_COMMAND;

    QString command = argumentList[0].toUpper();

    if (command == "QUIT" || command == "EXIT")
    {
        *isCommandFound = true;
        return cmdExit(myProject);
    }
    else if (command == "DEM" || command == "LOADDEM")
    {
        *isCommandFound = true;
        return cmdLoadDEM(myProject, argumentList);
    }
    else if (command == "POINT" || command == "LOADPOINTS")
    {
        *isCommandFound = true;
        return cmdOpenDbPoint(myProject, argumentList);
    }
    else if (command == "GRID" || command == "LOADGRID")
    {
        *isCommandFound = true;
        return cmdLoadMeteoGrid(myProject, argumentList);
    }
    else if (command == "LOG" || command == "SETLOGFILE")
    {
        *isCommandFound = true;
        return cmdSetLogFile(myProject, argumentList);
    }
    else if (command == "DAILYCSV" || command == "EXPORTDAILYDATACSV")
    {
        *isCommandFound = true;
        return cmdExportDailyDataCsv(myProject, argumentList);
    }
    else
    {
        // TODO:
        // other shared commands
    }

    return PRAGA_INVALID_COMMAND;
}

