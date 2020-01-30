#ifndef SHELL_H
#define SHELL_H

    #include <string>
    class QString;
    class QStringList;
    class Project;

    bool attachOutputToConsole();
    void openNewConsole();
    void sendEnterKey(void);
    bool isConsoleForeground();

    QString getTimeStamp(QStringList argumentList);
    QStringList getArgumentList(QString commandLine);
    QString getCommandLine(QString programName);
    QStringList getSharedCommandList();

    bool executeSharedCommand(Project* myProject, QStringList argumentList, bool *isCommandFound);

    bool cmdLoadDEM(Project* myProject, QStringList argumentList);
    bool cmdLoadMeteoGrid(Project* myProject, QStringList argumentList);
    bool cmdSetLogFile(Project* myProject, QStringList argumentList);
    bool cmdExit(Project* myProject);


#endif // SHELL_H
