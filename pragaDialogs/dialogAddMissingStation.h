#ifndef DIALOGADDMISSINGSTATION_H
#define DIALOGADDMISSINGSTATION_H

#include <QtWidgets>

class DialogAddMissingStation : public QDialog
{
    Q_OBJECT
private:
    QList<QString> idStations;
    QList<QString> nameStations;
    QListWidget* listStations;

public:
    DialogAddMissingStation(QList<QString> idStations, QList<QString> nameStations);
    ~DialogAddMissingStation();
    void done(bool res);
    QList<QString> getSelectedStations();
};

#endif // DIALOGADDMISSINGSTATION_H
