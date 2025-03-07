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

        QRadioButton pointsButton;
        QRadioButton gridButton;

        bool isMeteoGrid;
        bool isAnomaly;
        bool saveClima;

        QComboBox variableList;

        QDateEdit currentDay;
        QLabel currentDayLabel;

        QLineEdit firstYearEdit;
        QLineEdit lastYearEdit;
        QLineEdit offsetEdit;

        QLabel genericStartLabel;
        QLabel genericEndLabel;
        QLabel nrYearLabel;
        QLabel dailyOffsetLabel;
        QLabel offsetDateDisplayLabel;

        QDateEdit genericPeriodStart;
        QDateEdit genericPeriodEnd;
        QLineEdit dailyOffset;
        QDateEdit offsetDateDisplay;

        QLineEdit nrYear;
        QCheckBox readParam;
        QCheckBox dailyCumulated;

        QComboBox periodTypeList;
        QComboBox elaborationList;
        QComboBox secondElabList;
        QLineEdit periodDisplay;

        QLineEdit elab1Parameter;
        QLineEdit elab2Parameter;

        DialogAnomaly anomaly;
        QPushButton copyData;

        QPushButton addClimate;
        QPushButton delClimate;
        QPushButton loadXML;
        QPushButton appendXML;
        QPushButton delAll;

        SaveClimaLayout saveClimaLayout;

        QList<QString> climateDbElab;
        QComboBox climateDbElabList;


    public:
        DialogMeteoComputation(QSettings *settings, bool isMeteoGridLoaded, bool isMeteoPointLoaded, bool isAnomaly, bool isSaveClima);
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

        void copyDataFromXML();
        void saveDataToXML();
        void targetChange();
        void changeOffsetDate();

        bool getIsMeteoGrid() const { return isMeteoGrid; }
        QList<QString> getElabSaveList() { return saveClimaLayout.getList(); }

};


#endif // DIALOGMETEOCOMPUTATION_H
