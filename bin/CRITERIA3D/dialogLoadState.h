#ifndef DIALOGLOADSTATE_H
#define DIALOGLOADSTATE_H

#include <QtWidgets>

class DialogLoadState : public QDialog
{
    Q_OBJECT
private:
    QComboBox stateListComboBox;
public:
    DialogLoadState(QList<QString> allStates);
    QString getSelectedState();
};

#endif // DIALOGLOADSTATE_H
