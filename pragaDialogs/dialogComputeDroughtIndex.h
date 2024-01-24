#ifndef DIALOGCOMPUTEDROUGHTINDEX_H
#define DIALOGCOMPUTEDROUGHTINDEX_H

#include <QtWidgets>
#include "meteo.h"

class DialogComputeDroughtIndex : public QDialog
{
    Q_OBJECT
public:
    DialogComputeDroughtIndex(bool isMeteoGridLoaded, bool isMeteoPointLoaded, int yearPointsFrom, int yearPointsTo, int yearGridFrom, int yearGridTo);
    ~DialogComputeDroughtIndex();

    void indexClicked(QListWidgetItem* item);
    void done(bool res);
    QString getIndex() const;
    int getYearFrom() const;
    int getYearTo() const;

private:
    bool isMeteoPointLoaded;
    bool isMeteoGridLoaded;
    bool isMeteoGrid;
    QRadioButton pointsButton;
    QRadioButton gridButton;
    QListWidget listIndex;
    QLineEdit yearFrom;
    QLineEdit yearTo;
};

#endif // DIALOGCOMPUTEDROUGHTINDEX_H
