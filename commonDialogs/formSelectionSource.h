#ifndef FORMSELECTIONSOURCE_H
#define FORMSELECTIONSOURCE_H
/*
#include <QtWidgets>
*/
#include <QWidget>


class FormSelectionSource : public QWidget
{
    Q_OBJECT

public:
    FormSelectionSource();
    void sourceDone(int res);
    QString getSourceSelection();
    int getSourceSelectionId();
    void sourceChange();

private:
    /*
    QGroupBox *SourceSelection();
*/
};

#endif // FORMSELECTIONSOURCE_H
