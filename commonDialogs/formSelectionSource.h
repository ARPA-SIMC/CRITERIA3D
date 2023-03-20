#ifndef FORMSELECTIONSOURCE_H
#define FORMSELECTIONSOURCE_H

#include <QtWidgets>

//#include <QWidget>


class FormSelectionSource : public QDialog
{
    Q_OBJECT

public:
    FormSelectionSource();
    void done(int res);
    //QString getSourceSelection();
    int getSourceSelectionId();
    //void sourceChange();

private:
    QRadioButton* pointButton;
    QRadioButton* gridButton;
    /*
    QGroupBox *SourceSelection();
*/
};

#endif // FORMSELECTIONSOURCE_H
