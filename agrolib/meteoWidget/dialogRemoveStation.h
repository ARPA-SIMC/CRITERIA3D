#ifndef DIALOGREMOVESTATION_H
#define DIALOGREMOVESTATION_H

#include <QtWidgets>

class DialogRemoveStation : public QDialog
{
    Q_OBJECT

private:
    QList<QString> allStations;
    QList<QString> selectedStations;
    QListWidget* listAllStations;

public:
    DialogRemoveStation(QList<QString> allStations);
    QList<QString> getSelectedStations();
    void done(bool res);
};

#endif // DIALOGREMOVESTATION_H
