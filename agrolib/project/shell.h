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

    int executeSharedCommand(Project* myProject, QList<QString> argumentList, bool *isCommandFound);

    int cmdLoadDEM(Project* myProject, QList<QString> argumentList);
    int cmdLoadMeteoGrid(Project* myProject, QList<QString> argumentList);
    int cmdSetLogFile(Project* myProject, QList<QString> argumentList);
    int cmdExit(Project* myProject);


#endif // SHELL_H
