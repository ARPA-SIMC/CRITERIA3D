#ifndef DIALOGREMOVESTATION_H
#define DIALOGREMOVESTATION_H

#include <QtWidgets>

class DialogRemoveStation : public QDialog
{
    Q_OBJECT

private:
    QList<QString> _activeStationsList;
    QList<QString> _selectedStations;
    QListWidget* _listStationsWidget;

public:
    DialogRemoveStation(QList<QString> allStations);
    QList<QString> getSelectedStations();
    void done(bool res);
};

#endif // DIALOGREMOVESTATION_H
