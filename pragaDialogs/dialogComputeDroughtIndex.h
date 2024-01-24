#ifndef DIALOGCOMPUTEDROUGHTINDEX_H
#define DIALOGCOMPUTEDROUGHTINDEX_H

#include <QtWidgets>
#include "meteo.h"

class DialogComputeDroughtIndex : public QDialog
{
    Q_OBJECT
public:
    DialogComputeDroughtIndex(bool isMeteoGridLoaded, bool isMeteoPointLoaded, QDate myDatePointsFrom, QDate myDatePointsTo, QDate myDateGridFrom, QDate myDateGridTo);
    ~DialogComputeDroughtIndex();

    void indexClicked(QListWidgetItem* item);
    void done(bool res);
    QString getIndex() const;
    QDate getDateFrom() const;
    QDate getDateTo() const;

private:
    bool isMeteoPointLoaded;
    bool isMeteoGridLoaded;
    bool isMeteoGrid;
    QRadioButton pointsButton;
    QRadioButton gridButton;
    QListWidget listIndex;
    QDateEdit dateFrom;
    QDateEdit dateTo;
};

#endif // DIALOGCOMPUTEDROUGHTINDEX_H
