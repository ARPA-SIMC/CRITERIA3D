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

    QComboBox* _listActiveStationsWidget;
    QLineEdit* _singleValueEdit;
    Crit3DMeteoPoint* _allMeteoPointsPointer;
    QListWidget* _listNearStationsWidget;

    int _nrAllMeteoPoints;

public:
    DialogAddStation(const QList<QString> &activeStationsList, Crit3DMeteoPoint *allMeteoPointsPointer, int nrAllMeteoPoints);

    double getSingleValue();
    void searchStations();

};

#endif // DIALOGADDSTATION_H
