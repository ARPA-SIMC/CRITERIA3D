#include "console.h"
#include <iostream>
#include <sstream>

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

