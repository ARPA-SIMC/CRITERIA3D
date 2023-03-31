#ifndef FORMSELECTION_H
#define FORMSELECTION_H

    #include <QtWidgets>

    class FormSelection : public QDialog
    {
        Q_OBJECT

    private:
        QList<QString> stringList;
        QComboBox* cmbStringList;
        void done(int res);

    public:
        FormSelection(QList<QString> stringList_);

        QString getSelection();
        int getSelectionId();
    };

#endif // FORMSELECTION_H
