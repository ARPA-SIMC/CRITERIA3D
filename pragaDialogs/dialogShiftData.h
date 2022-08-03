#ifndef DIALOGSHIFTDATA_H
#define DIALOGSHIFTDATA_H

#include <QtWidgets>

class DialogShiftData : public QDialog
{
    Q_OBJECT
public:
    DialogShiftData(QDate myDate);
    ~DialogShiftData();
    void done(bool res);
    int getShift() const;

private:
    QLineEdit shiftEdit;
    QComboBox variable;
    QDateEdit dateFrom;
    QDateEdit dateTo;
};

#endif // DIALOGSHIFTDATA_H
