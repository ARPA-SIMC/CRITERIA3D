#ifndef FORMSELECTION_H
#define FORMSELECTION_H

#include <QtWidgets>

class FormSelection : public QDialog
{
    Q_OBJECT

private:
    QList<QString> stringList;
    QComboBox* cmbStringList;

public:
    FormSelection(QList<QString> stringList_);
    ~FormSelection();
    void done(int res);
    QString getSelection();
    int getSelectionId();
};

#endif // FORMSELECTION_H
