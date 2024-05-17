#ifndef DIALOGSELECTWELL_H
#define DIALOGSELECTWELL_H

#include <QtWidgets>

class DialogSelectWell : public QDialog
{
    public:
        DialogSelectWell(QList<QString> wellsId);
        QString getIdSelected() const;
        void done(bool res);

private:
        QList<QString> wellsId;
        QString idSelected;
        QComboBox* buttonWells;
};

#endif // DIALOGSELECTWELL_H
