#include "shell.h"
#include "project.h"
#include <iostream>
#include <sstream>
#include <QString>
#include <QStringList>

#ifdef _WIN32
    #include "Windows.h"
    #pragma comment(lib, "User32.lib")
#endif

using namespace std;


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


QString getTimeStamp(QStringList argumentList)
{
    QString myString = ">> ";
    myString += QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    myString += " >>";

    for (int i = 0; i < argumentList.size(); i++)
    {
        myString += " " + argumentList[i];
    }

    return myString;
}


QStringList getArgumentList(QString commandLine)
{
    string str;
    QStringList argumentList;

    istringstream stream(commandLine.toStdString());
    while (stream >> str)
    {
        argumentList.append(QString::fromStdString(str));
    }

    return argumentList;
}


QString getCommandLine(QString programName)
{
    string commandLine;

    cout << programName.toStdString() << ">";
    getline (cin, commandLine);

    return QString::fromStdString(commandLine);
}


QStringList getSharedCommandList()
{
    QStringList cmdList;

    cmdList.append("Log     | SetLogFile");
    cmdList.append("DEM     | LoadDEM");
    cmdList.append("GRID    | LoadGrid");
    cmdList.append("Quit    | Exit");

    return cmdList;
}


bool cmdExit(Project* myProject)
{
    myProject->requestedExit = true;

    // TODO: close project

    return true;
}


bool cmdLoadDEM(Project* myProject, QStringList argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing DEM file name.");
        // TODO: USAGE
        return false;
    }
    else
    {
        return myProject->loadDEM(argumentList[1]);
    }
}


bool cmdLoadMeteoGrid(Project* myProject, QStringList argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing Grid file name.");
        // TODO: USAGE
        return false;
    }
    else
    {
        if (!myProject->loadMeteoGridDB(argumentList[1]))
        {
            return false;
        }
        else
        {
            myProject->meteoGridDbHandler->meteoGrid()->createRasterGrid();
            return true;
        }
    }
}


bool cmdSetLogFile(Project* myProject, QStringList argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing Log file name.");
        // TODO: USAGE
        return false;
    }
    else
    {
        return myProject->setLogFile(argumentList[1]);
    }
}


bool executeSharedCommand(Project* myProject, QStringList argumentList, bool* isCommandFound)
{
    *isCommandFound = false;
    if (argumentList.size() == 0) return false;

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
    else
    {
        // TODO:
        // other shared commands
    }

    return false;
}

