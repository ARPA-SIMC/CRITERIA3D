#ifndef DIALOGADDSTATION_H
#define DIALOGADDSTATION_H

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
    QComboBox* _listNearStationsWidget; //non so se serve, forse come lista di risultato sì

public:
    DialogAddStation(QList<QString> _activeStationsList);
    QList<QString> getSelectedStations();
    void searchStations(bool res);
    double getSingleValue();
};

#endif // DIALOGADDSTATION_H
