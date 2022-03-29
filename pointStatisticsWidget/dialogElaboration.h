#ifndef DIALOGELABORATION_H
#define DIALOGELABORATION_H

#include <QString>
#include <QSettings>
#include <QGridLayout>
#include <QComboBox>
#include <QtWidgets>
#include "crit3dClimate.h"

class DialogElaboration : public QDialog
{
    Q_OBJECT

public:
    DialogElaboration(QSettings *settings, Crit3DClimate *clima, QDate firstDate, QDate lastDate);
    void done(bool res);
    void displayPeriod(const QString value);
    void listElaboration(const QString value);
    void changeElab(const QString value);
    bool checkValidData();
private:
    QSettings *settings;
    Crit3DClimate *clima;
    QDate firstDate;
    QDate lastDate;
    QDateEdit currentDay;
    QLabel currentDayLabel;
    QComboBox variableList;
    QLineEdit firstYearEdit;
    QLineEdit lastYearEdit;
    QLabel genericStartLabel;
    QLabel genericEndLabel;
    QLabel nrYearLabel;
    QDateEdit genericPeriodStart;
    QDateEdit genericPeriodEnd;
    QLineEdit nrYear;
    QComboBox periodTypeList;
    QComboBox elaborationList;
    QLineEdit periodDisplay;
    QLineEdit elab1Parameter;

};

#endif // DIALOGELABORATION_H
