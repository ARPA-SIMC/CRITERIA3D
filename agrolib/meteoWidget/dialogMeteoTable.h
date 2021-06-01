#ifndef DIALOGMETEOTABLE_H
#define DIALOGMETEOTABLE_H

#include "meteoPoint.h"
#include "meteoTable.h"
#include <QtWidgets>

class DialogMeteoTable : public QDialog
{
    Q_OBJECT

    private:

    Crit3DMeteoSettings* meteoSettings;
    QVector<Crit3DMeteoPoint> meteoPoints;
    QDate firstDate;
    QDate lastDate;
    frequencyType currentFreq;
    QStringList currentVariables;
    MeteoTable* meteoTable;
    QStringList labels;
    QStringList meteoTableHeader;

    public:
        DialogMeteoTable(Crit3DMeteoSettings* meteoSettings_, QVector<Crit3DMeteoPoint> meteoPoints, QDate firstDate, QDate lastDate, frequencyType currentFreq, QStringList currentVariables);
        ~DialogMeteoTable();
};

#endif // DIALOGMETEOTABLE_H
