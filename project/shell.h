#ifndef SHELL_H
#define SHELL_H

    #include <string>
    class QString;
    template <typename T> class QList;
    class Project;

    bool attachOutputToConsole();
    void openNewConsole();
    void sendEnterKey(void);
    bool isConsoleForeground();

    QString getTimeStamp(QList<QString> argumentList);
    QList<QString> getArgumentList(QString commandLine);
    QString getCommandLine(QString programName);
    QList<QString> getSharedCommandList();

    bool executeSharedCommand(Project* myProject, QList<QString> argumentList, bool *isCommandFound);

    bool cmdLoadDEM(Project* myProject, QList<QString> argumentList);
    bool cmdLoadMeteoGrid(Project* myProject, QList<QString> argumentList);
    bool cmdSetLogFile(Project* myProject, QList<QString> argumentList);
    bool cmdExit(Project* myProject);


#endif // SHELL_H
