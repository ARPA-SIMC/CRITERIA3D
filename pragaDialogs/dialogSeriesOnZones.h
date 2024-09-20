#ifndef DIALOGSERIESONZONES_H
#define DIALOGSERIESONZONES_H

#include <QtWidgets>

#include "meteo.h"

class DialogSeriesOnZones: public QDialog
{
    Q_OBJECT

    private:
        QSettings* settings;
        QList<QString> aggregations;
        QComboBox variableList;
        QLabel genericStartLabel;
        QLabel genericEndLabel;
        QDateEdit genericPeriodStart;
        QDateEdit genericPeriodEnd;
        QComboBox spatialElab;

        meteoVariable variable;
        QDate startDate;
        QDate endDate;
        QString spatialElaboration;

    public:
        DialogSeriesOnZones(QSettings *settings, QList<QString> aggregations, QDate currentDate, bool isHourly);

        void done(bool res);
        bool checkValidData();

        meteoVariable getVariable() const
        { return variable; }

        QDate getStartDate() const
        { return startDate; }

        QDate getEndDate() const
        { return endDate; }

        QString getSpatialElaboration() const
        { return spatialElaboration; }

};


#endif // DIALOGSERIESONZONES_H
