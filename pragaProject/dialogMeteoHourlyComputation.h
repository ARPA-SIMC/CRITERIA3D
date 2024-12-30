#ifndef DIALOGMETEOHOURLYCOMPUTATION_H
#define DIALOGMETEOHOURLYCOMPUTATION_H

#include <QString>
#include <QSettings>
#include <QDialog>
#include <QComboBox>
#include <QSpinBox>
#include <QRadioButton>
#include <QDateEdit>


class DialogMeteoHourlyComputation : public QDialog
{
    Q_OBJECT

    private:
        bool isMeteoGrid;
        QSettings* settings;

        QRadioButton pointsButton;
        QRadioButton gridButton;

        QComboBox variableList;
        QComboBox elaborationList;

        QDateEdit timeRangeStart;
        QDateEdit timeRangeEnd;
        QSpinBox hourStart;
        QSpinBox hourEnd;

    public:
        DialogMeteoHourlyComputation(QSettings *settings, bool isMeteoGridLoaded, bool isMeteoPointLoaded);
        void done(bool res);

        void targetChange();

        void listElaboration(const QString variable);

        bool getIsMeteoGrid() const { return isMeteoGrid; }
};


#endif // DIALOGMETEOHOURLYCOMPUTATION_H
