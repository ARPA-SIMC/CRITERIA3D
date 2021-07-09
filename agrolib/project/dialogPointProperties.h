#ifndef DIALOGPOINTPROPERTIES_H
#define DIALOGPOINTPROPERTIES_H

#include <QString>
#include <QGridLayout>
#include <QtWidgets>

class DialogPointProperties : public QDialog
{
    Q_OBJECT

public:
    DialogPointProperties(QList<QString> pragaProperties, QList<QString> CSVFields);
    void propertiesClicked(QListWidgetItem* item);
    void csvClicked(QListWidgetItem* item);
    void joinedClicked(QListWidgetItem* item);
    void addCouple();
    void deleteCouple();
    QList<QString> getJoinedList();
    void done(bool res);

private:
    QList<QString> pragaProperties;
    QList<QString> CSVFields;
    QListWidget* propertiesList;
    QListWidget* csvList;
    QListWidget* joinedList;
    QPushButton* joinButton;
    QPushButton* deleteButton;
    QString propertiesSelected;
    QString csvSelected;

};

#endif // DIALOGPOINTPROPERTIES_H
