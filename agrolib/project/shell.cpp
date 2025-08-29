#include "commonConstants.h"
#include "meteo.h"
#include "project.h"
#include "shell.h"
#include "dbMeteoGrid.h"
#include "utilities.h"

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


bool closeConsole()
{
    #ifdef _WIN32
        FreeConsole();
    #endif

    return true;
}


bool isConsoleForeground()
{
    #ifdef _WIN32
        return (GetConsoleWindow() == GetForegroundWindow());
    #endif

    return true;
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


QString getTimeStamp(const QList<QString> &argumentList)
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


QList<QString> getArgumentList(const QString &commandLine)
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


QString getCommandLine(const QString &programName)
{
    std::string commandLine;

    std::cout.flush() << programName.toStdString() << ">";
    std::getline(std::cin, commandLine);

    return QString::fromStdString(commandLine);
}


QList<QString> getSharedCommandList()
{
    QList<QString> cmdList;

    cmdList.append("DEM             | LoadDEM");
    cmdList.append("Point           | LoadPoints");
    cmdList.append("Grid            | LoadGrid");
    cmdList.append("Log             | SetLogFile");
    cmdList.append("Quit            | Exit");
    cmdList.append("DailyCsv        | ExportDailyDataCsv");

    return cmdList;
}


int cmdExit(Project* myProject)
{
    myProject->requestedExit = true;
    return PRAGA_OK;
}


int cmdLoadDEM(Project* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->errorString = "Missing DEM file name.";
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
        myProject->errorString = "Missing db point name";
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
        myProject->errorString = "Missing grid file name";
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
        myProject->errorString = "Missing log file name";
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
    // GA questa funzione scrive degli errori ed esce, ma ritorna sempre PRAGA_OK. e' giusto?
    QString outputPath = myProject->getProjectPath() + PATH_OUTPUT;

    if (argumentList.size() < 2)
    {
        QString usage = "Usage:\n"
                        "ExportDailyDataCsv -v:variableList [-TPREC] [-t:type] -d1:firstDate [-d2:lastDate] [-l:idList] [-p:outputPath]\n"
                        "-v         list of comma separated variables (varname: TMIN, TMAX, TAVG, PREC, RHMIN, RHMAX, RHAVG, RAD, ET0_HS, ET0_PM, LEAFW) \n"
                        "-TPREC     export Tmin, Tmax, Tavg, Prec \n"
                        "-t         type: GRID|POINTS (default: GRID) \n"
                        "-d1, -d2   date format: YYYY-MM-DD (default: lastDate = yesterday) \n"
                        "-l         list of output points or cells filename  (default: ALL active cells/points) \n"
                        "-p         output Path (default: " + outputPath + ") \n";
        myProject->logInfo(usage);
        return PRAGA_OK;
    }

    QString typeStr = "GRID";
    QDate firstDate, lastDate;
    QList<meteoVariable> variableList;
    bool isTPrec = false;
    QString idListFileName = "";

    for (int i = 1; i < argumentList.size(); i++)
    {
        if (argumentList.at(i).left(6).toUpper() == "-TPREC")
        {
            isTPrec = true;
            variableList = {dailyAirTemperatureMin, dailyAirTemperatureMax, dailyAirTemperatureAvg, dailyPrecipitation};
        }

        // variables list
        if (argumentList.at(i).left(3) == "-v:")
        {
            QString variables = argumentList[i].right(argumentList[i].length()-3).toUpper();
            QList<QString> varNameList = variables.split(",");
            for (int i = 0; i < varNameList.size(); i++)
            {
                std::string varString = "DAILY_" + varNameList[i].toStdString();
                meteoVariable var = getMeteoVar(varString);
                if (var != noMeteoVar)
                {
                    variableList.append(var);
                }
                else
                {
                    myProject->logError("Wrong variable: " + varNameList[i]);
                    return PRAGA_OK;
                }
            }
        }

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
                myProject->logError("Wrong type: available GRID or POINTS.");
                return PRAGA_OK;
            }
        }

        if (argumentList.at(i).left(3) == "-l:")
        {
            idListFileName = argumentList[i].right(argumentList[i].length()-3);
            idListFileName = myProject->getCompleteFileName(idListFileName, PATH_OUTPUT);
        }

        if (argumentList.at(i).left(3) == "-p:" || argumentList.at(i).left(3) == "-o:")
        {
            outputPath = argumentList[i].right(argumentList[i].length()-3);
            if (outputPath.size() > 0)
            {
                if (outputPath.at(0) == '.')
                {
                    QString completeOutputPath = myProject->getProjectPath() + outputPath;
                    outputPath = QDir().cleanPath(completeOutputPath);
                }
                else
                {
                    if(getFileName(outputPath) == outputPath)
                    {
                        QString completeOutputPath = myProject->getProjectPath() + PATH_OUTPUT + outputPath;
                        outputPath = QDir().cleanPath(completeOutputPath);
                    }
                }
            }
        }
    }

    // check first date (mandatory)
    if (! firstDate.isValid())
    {
        myProject->logError("Missing first date: use -d1:firstDate");
        return PRAGA_OK;
    }

    // check last date (default: yesterday)
    if (! lastDate.isValid())
    {
        lastDate = QDateTime::currentDateTime().date().addDays(-1);
    }

    myProject->logInfo("... first date is: " + firstDate.toString());
    myProject->logInfo("... last date is: " + lastDate.toString());


    if (variableList.isEmpty())
    {
        myProject->logError("Missing variables.");
        return PRAGA_OK;
    }

    if (idListFileName != "")
    {
        myProject->logInfo("... ID list file is: " + idListFileName);
    }
    else
    {
        if (typeStr == "GRID")
            myProject->logInfo("... export ALL meteoGrid cells");
        else
            myProject->logInfo("... export ALL meteo points");
    }

    myProject->logInfo("... output path is: " + outputPath);

    if (typeStr == "GRID")
    {
        if (! myProject->meteoGridLoaded)
        {
            myProject->logError("No meteo grid open.");
            return PRAGA_ERROR;
        }

        if (! myProject->meteoGridDbHandler->exportDailyDataCsv(variableList, firstDate, lastDate,
                                                               idListFileName, outputPath, myProject->errorString))
        {
            myProject->logError();
            return PRAGA_ERROR;
        }
    }
    else if (typeStr == "POINTS")
    {
        if (! myProject->exportMeteoPointsDailyDataCsv(isTPrec, firstDate, lastDate, idListFileName, outputPath))
        {
            myProject->logError();
            return PRAGA_ERROR;
        }
    }

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

    return NOT_SHARED_COMMAND;
}

