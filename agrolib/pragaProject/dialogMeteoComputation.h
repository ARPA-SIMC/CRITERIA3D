#ifndef DIALOGMETEOCOMPUTATION_H
#define DIALOGMETEOCOMPUTATION_H

#include <QString>
#include <QSettings>
#include <QGridLayout>
#include <QComboBox>
#include <QtWidgets>

#include "dialogAnomaly.h"
#include "saveClimaLayout.h"


class DialogMeteoComputation : public QDialog
{

    Q_OBJECT

    private:
        QSettings* settings;
        bool isMeteoGrid;
        bool isAnomaly;
        bool saveClima;
        QString title;
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
        QCheckBox readParam;

        QComboBox periodTypeList;
        QComboBox elaborationList;
        QComboBox secondElabList;
        QLineEdit periodDisplay;

        QLineEdit elab1Parameter;
        QLineEdit elab2Parameter;

        DialogAnomaly anomaly;
        QPushButton copyData;

        QPushButton add;
        QPushButton del;
        QPushButton loadXML;
        QPushButton appendXML;
        QPushButton delAll;
        SaveClimaLayout saveClimaLayout;

        QList<QString> climateDbElab;
        QComboBox climateDbElabList;



    public:
        DialogMeteoComputation(QSettings *settings, bool isMeteoGrid, bool isAnomaly, bool saveClima);
        void done(bool res);
        void displayPeriod(const QString value);
        void checkYears();
        void listElaboration(const QString value);
        void listSecondElab(const QString value);
        void activeSecondParameter(const QString value);
        void readParameter(int state);
        void copyDataToAnomaly();
        void copyDataToSaveLayout();
        bool checkValidData();
        QList<QString> getElabSaveList();
        void copyDataFromXML();
        void saveDataToXML();
};


#endif // DIALOGMETEOCOMPUTATION_H
