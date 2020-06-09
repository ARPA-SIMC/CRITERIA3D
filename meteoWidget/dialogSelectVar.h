#ifndef DIALOGSELECTVAR_H
#define DIALOGSELECTVAR_H

#include <QtWidgets>

class DialogSelectVar : public QDialog
{
    Q_OBJECT

private:
    QStringList allVar;
    QStringList selectedVar;
    QListWidget* listAllVar;
    QListWidget* listSelectedVar;
    QPushButton *addButton;
    QPushButton *deleteButton;

public:
    DialogSelectVar(QStringList allVar, QStringList selectedVar);
    void variableAllClicked(QListWidgetItem* item);
    void variableSelClicked(QListWidgetItem* item);
    void addVar();
    void deleteVar();
    QStringList getSelectedVariables();
};

#endif // DIALOGSELECTVAR_H
