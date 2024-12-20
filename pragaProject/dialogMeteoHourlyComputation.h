#ifndef DIALOGMETEOHOURLYCOMPUTATION_H
#define DIALOGMETEOHOURLYCOMPUTATION_H

#include <QString>
#include <QSettings>
#include <QComboBox>
#include <QtWidgets>


class DialogMeteoHourlyComputation : public QDialog
{
    Q_OBJECT

    private:
        QSettings* settings;

        QRadioButton pointsButton;
        QRadioButton gridButton;

        bool isMeteoGrid;

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

        QLineEdit elab1Parameter;


    public:
        DialogMeteoHourlyComputation(QSettings *settings, bool isMeteoGridLoaded, bool isMeteoPointLoaded);
        void done(bool res);

        void displayPeriod(const QString value);
        void listElaboration(const QString value);

        bool checkValidData();
        void targetChange();

        bool getIsMeteoGrid() const { return isMeteoGrid; }
};


#endif // DIALOGMETEOHOURLYCOMPUTATION_H
