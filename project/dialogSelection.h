#ifndef DIALOGSELECTION_H
#define DIALOGSELECTION_H

    #include <QString>
    #include <QDateTime>
    #include "meteo.h"
    #include "project.h"

    class Project;

    QString editValue(QString windowsTitle, QString defaultValue);

    meteoVariable chooseColorScale();
    frequencyType chooseFrequency(Project *project_);
    meteoVariable chooseMeteoVariable(Project *project_);


#endif // DIALOGSELECTION_H
