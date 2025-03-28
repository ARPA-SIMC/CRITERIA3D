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
    QLineEdit _singleValueEdit;
    Crit3DMeteoPoint* _allMeteoPointsPointer;
    int _nrAllMeteoPoints;
    QListWidget* _listNearStationsWidget;
    QVector<Crit3DMeteoPoint> _meteoPoints;

public:
    DialogAddStation(QList<QString> _activeStationsList, Crit3DMeteoPoint *allMeteoPointsPointer, QVector<Crit3DMeteoPoint> _meteoPoints);
    double getSingleValue();
    void searchStations();

};

#endif // DIALOGADDSTATION_H
