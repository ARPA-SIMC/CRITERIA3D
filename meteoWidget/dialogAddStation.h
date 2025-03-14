#ifndef DIALOGADDSTATION_H
#define DIALOGADDSTATION_H

#include "meteoPoint.h"
#include <QtWidgets>

class DialogAddStation : public QDialog
{
    Q_OBJECT

private:
    QList<QString> _activeStationsList;
    QList<QString> _selectedStations;
    QList<QString> _nearStationsList;

    QListWidget* _listActiveStationsWidget;
    QLineEdit _singleValueEdit;  //per scegliere la distanza
    Crit3DMeteoPoint* _allMeteoPointsPointer;
    QListWidget* _listNearStationsWidget;

public:
    DialogAddStation(QList<QString> _activeStationsList);
    double getSingleValue();
    void searchStations(bool res, Crit3DMeteoPoint* _allMeteoPointsPointer, int nrMeteoPoints);

};

#endif // DIALOGADDSTATION_H
