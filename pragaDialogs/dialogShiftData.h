#ifndef DIALOGSHIFTDATA_H
#define DIALOGSHIFTDATA_H

#include <QtWidgets>
#include "meteo.h"

class DialogShiftData : public QDialog
{
    Q_OBJECT
public:
    DialogShiftData(QDate myDate, bool allPoints);
    ~DialogShiftData();
    void done(bool res);
    int getShift() const;
    meteoVariable getVariable() const;
    QDate getDateFrom() const;
    QDate getDateTo() const;

private:
    QLineEdit shiftEdit;
    QComboBox variable;
    QDateEdit dateFrom;
    QDateEdit dateTo;
};

#endif // DIALOGSHIFTDATA_H
