#ifndef DIALOGADDREMOVEDATASET_H
#define DIALOGADDREMOVEDATASET_H

#include <QtWidgets>

class DialogAddRemoveDataset : public QDialog
{
    Q_OBJECT

private:
    QList<QString> availableDataset;
    QList<QString> dbDataset;
    QListWidget* listAvailableDataset;
    QListWidget* listDbDataset;
    QPushButton *addButton;
    QPushButton *deleteButton;

public:
    DialogAddRemoveDataset(QList<QString> availableDataset, QList<QString> dbDataset);
    void datasetAllClicked(QListWidgetItem* item);
    void datasetDbClicked(QListWidgetItem* item);
    void addDataset();
    void deleteDataset();
    QList<QString> getDatasetDb();
};

#endif // DIALOGADDREMOVEDATASET_H
