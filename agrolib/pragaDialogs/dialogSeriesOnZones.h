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

        bool isHourly_;
        meteoVariable variable_;
        QDate startDate_;
        QDate endDate_;
        QString spatialElaboration_;

    public:
        DialogSeriesOnZones(QSettings *settings, QList<QString> aggregations, QDate currentDate, bool isHourly);

        void done(bool res);
        bool checkValidData();

        meteoVariable getVariable() const
        { return variable_; }

        QDate getStartDate() const
        { return startDate_; }

        QDate getEndDate() const
        { return endDate_; }

        QString getSpatialElaboration() const
        { return spatialElaboration_; }

};


#endif // DIALOGSERIESONZONES_H
