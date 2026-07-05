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
    QListWidget* _listNearStationsWidget;
    std::vector<Crit3DMeteoPoint> _allMeteoPoints;

public:
    DialogAddStation(const QList<QString> &activeStationsList, const std::vector<Crit3DMeteoPoint> &allMeteoPoints);

    double getSingleValue();
    void searchStations();

};

#endif // DIALOGADDSTATION_H
