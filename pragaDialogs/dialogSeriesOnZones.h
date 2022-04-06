#ifndef DIALOGSERIESONZONES_H
#define DIALOGSERIESONZONES_H

#include <QString>
#include <QSettings>
#include <QGridLayout>
#include <QComboBox>

#include <QtWidgets>
#include "meteoGrid.h"

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
        DialogSeriesOnZones(QSettings *settings, QList<QString> aggregations, QDate currentDate);
        void done(bool res);
        bool checkValidData();

        meteoVariable getVariable() const;
        QDate getStartDate() const;
        QDate getEndDate() const;
        QString getSpatialElaboration() const;
};


#endif // DIALOGSERIESONZONES_H
