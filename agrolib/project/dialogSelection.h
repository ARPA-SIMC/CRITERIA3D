#ifndef DIALOGSELECTION_H
#define DIALOGSELECTION_H

    #include <QString>
    #include <QDateTime>
    #include "meteo.h"
    #include "project.h"

    class Project;

    QString editValue(QString windowsTitle, QString defaultValue);

    meteoVariable chooseColorScale();
    frequencyType chooseFrequency(const Project &myProject);
    meteoVariable chooseMeteoVariable(const Project &myProject);


#endif // DIALOGSELECTION_H
