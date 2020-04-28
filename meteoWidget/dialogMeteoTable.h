#ifndef DIALOGMETEOTABLE_H
#define DIALOGMETEOTABLE_H

#include "meteoPoint.h"
#include <QtWidgets>

class DialogMeteoTable : public QDialog
{
    Q_OBJECT

    private:

    QVector<Crit3DMeteoPoint> meteoPoints;
    QDate firstDate;
    QDate lastDate;
    frequencyType currentFreq;
    QStringList currentVariables;
    QTableWidget* meteoTable;
    QStringList labels;
    QStringList meteoTableHeader;

    public:
        DialogMeteoTable(QVector<Crit3DMeteoPoint> meteoPoints, QDate firstDate, QDate lastDate, frequencyType currentFreq, QStringList currentVariables);
        ~DialogMeteoTable();
};

#endif // DIALOGMETEOTABLE_H
