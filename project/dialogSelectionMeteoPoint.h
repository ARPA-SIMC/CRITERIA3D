#ifndef DIALOGSELECTIONMETEOPOINT_H
#define DIALOGSELECTIONMETEOPOINT_H

#include <QString>
#include <QtWidgets>
#include "dbMeteoPointsHandler.h"

class DialogSelectionMeteoPoint : public QDialog
{
    Q_OBJECT

private:
    bool isActive;
    bool isSelect;
    QList<QString> municipalityList;
    QList<QString> provinceList;
    QList<QString> regionList;
    QList<QString> stateList;
    QList<QString> datasetList;
    QComboBox selectionMode;
    QComboBox selectionOperation;
    QComboBox selectionItems;
    QTextEdit editItems;
    bool itemFromList;
    void selectionModeChanged();


public:
    DialogSelectionMeteoPoint(bool isActive, bool isSelect, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler);
    QString getSelection();
    QString getOperation();
    QString getItem();
    void done(bool res);
};

#endif // DIALOGSELECTIONMETEOPOINT_H
