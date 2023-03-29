#ifndef DIALOGSELECTDATASET_H
#define DIALOGSELECTDATASET_H

#include <QtWidgets>

class DialogSelectDataset : public QDialog
{
    Q_OBJECT

private:
    QList<QString> activeDataset;
    QListWidget* listDataset;

public:
    DialogSelectDataset(QList<QString> activeDataset);
    ~DialogSelectDataset();
    void done(bool res);
    QList<QString> getSelectedDatasets();
};

#endif // DIALOGSELECTDATASET_H
