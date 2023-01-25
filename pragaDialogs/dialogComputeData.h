#ifndef DIALOGCOMPUTEDATA_H
#define DIALOGCOMPUTEDATA_H

#include <QtWidgets>
#include "meteo.h"

class DialogComputeData : public QDialog
{
    Q_OBJECT
public:
    DialogComputeData(QDate myDateFrom, QDate myDateTo, bool isGrid, bool allPoints);
    ~DialogComputeData();
    void allVarClicked(int toggled);
    void varClicked(QListWidgetItem* item);
    void done(bool res);
    QList <meteoVariable> getVariables() const;
    QDate getDateFrom() const;
    QDate getDateTo() const;

private:
    bool isGrid;
    QListWidget listVariable;
    QCheckBox allVar;
    QDateEdit dateFrom;
    QDateEdit dateTo;
};

#endif // DIALOGCOMPUTEDATA_H
