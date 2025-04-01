#ifndef DIALOGSELECTVAR_H
#define DIALOGSELECTVAR_H

#include <QtWidgets>

class DialogSelectVar : public QDialog
{
    Q_OBJECT

private:
    QList<QString> allVar;
    QList<QString> selectedVar;
    QListWidget* listAllVar;
    QListWidget* listSelectedVar;
    QPushButton *addButton;
    QPushButton *deleteButton;

public:
    DialogSelectVar(QList<QString> allVar, QList<QString> selectedVar);
    void variableAllClicked(QListWidgetItem* item);
    void variableSelClicked(QListWidgetItem* item);
    void addVar();
    void deleteVar();
    QList<QString> getSelectedVariables();
};

#endif // DIALOGSELECTVAR_H
