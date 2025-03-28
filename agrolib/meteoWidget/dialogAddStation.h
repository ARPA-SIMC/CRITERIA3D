#ifndef DIALOGADDSTATION_H
#define DIALOGADDSTATION_H

#include "meteoPoint.h"
#include <QtWidgets>

class DialogAddStation : public QDialog
{
    Q_OBJECT

private:
    int _nrAllMeteoPoints;

    QList<QString> _activeStationsList;
    QList<QString> _selectedStations;
    QList<QString> _nearStationsList;

    QComboBox* _listActiveStationsWidget;
    QLineEdit* _singleValueEdit;
    Crit3DMeteoPoint* _allMeteoPointsPointer;
    QListWidget* _listNearStationsWidget;

public:
    DialogAddStation(const QList<QString> &_activeStationsList, Crit3DMeteoPoint *allMeteoPointsPointer, int nrAllMeteoPoints);

    double getSingleValue();
    void searchStations();

};

#endif // DIALOGADDSTATION_H
