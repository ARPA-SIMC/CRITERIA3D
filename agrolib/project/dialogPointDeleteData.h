#ifndef DIALOGPOINTDELETEDATA_H
#define DIALOGPOINTDELETEDATA_H

#include <QString>
#include <QtWidgets>
#include "meteo.h"

class DialogPointDeleteData : public QDialog
{
    Q_OBJECT

private:
    QListWidget dailyVar;
    QListWidget hourlyVar;
    QDateEdit firstDateEdit;
    QDateEdit lastDateEdit;

    QList<QString> varD;
    QList<QString> varH;

    QCheckBox allDaily;
    QCheckBox allHourly;

public:
    DialogPointDeleteData(QDate currentdate);
    void allDailyVarClicked(int toggled);
    void allHourlyVarClicked(int toggled);
    void dailyItemClicked(QListWidgetItem * item);
    void hourlyItemClicked(QListWidgetItem * item);
    QList<meteoVariable> getVarD() const;
    QList<meteoVariable> getVarH() const;
    QDate getFirstDate();
    QDate getLastDate();
    bool getAllDailyVar();
    bool getAllHourlyVar();
    void done(bool res);
};

#endif // DIALOGPOINTDELETEDATA_H
