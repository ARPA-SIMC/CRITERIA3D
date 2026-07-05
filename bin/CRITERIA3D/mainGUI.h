#ifndef MAINGUI_H
#define MAINGUI_H

    class QString;
    class Crit3DProject;

    bool checkEnvironmentGUI(QString criteria3dHome);
    QString searchDefaultPath(QString startPath);
    int mainGUI(int argc, char *argv[], QString criteria3dHome, Crit3DProject& myProject);

#endif // MAINGUI_H
