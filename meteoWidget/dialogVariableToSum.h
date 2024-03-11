#ifndef DIALOGVARIABLETOSUM_H
#define DIALOGVARIABLETOSUM_H

#include <QtWidgets>

class DialogVariableToSum : public QDialog
{
    Q_OBJECT

private:
    QList<QString> variableList;
    QList<QString> selectedVariable;
    QList<QCheckBox*> checkList;

public:
    DialogVariableToSum(QList<QString> variableList);
    QList<QString> getSelectedVariable();
    void done(bool res);
};

#endif // DIALOGVARIABLETOSUM_H
