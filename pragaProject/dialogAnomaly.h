#ifndef DIALOGANOMALY_H
#define DIALOGANOMALY_H

#include <QString>
#include <QSettings>
#include <QGridLayout>
#include <QComboBox>
#include <QtWidgets>


class DialogAnomaly : public QDialog
{

    Q_OBJECT

    private:
        QSettings* AnomalySettings;
        QLineEdit variableElab;

        QDateEdit currentDay;
        QLabel currentDayLabel;

        QLabel firstDateLabel;
        QLineEdit firstYearEdit;

        QCheckBox readReference;

        QLabel lastDateLabel;
        QLineEdit lastYearEdit;

        QLabel periodTypeLabel;
        QLabel genericStartLabel;
        QLabel genericEndLabel;
        QLabel nrYearLabel;
        QDateEdit genericPeriodStart;
        QDateEdit genericPeriodEnd;
        QLineEdit nrYear;
        QCheckBox readParam;

        QComboBox periodTypeList;
        QLabel elab;
        QComboBox elaborationList;
        QLabel secondElab;
        QComboBox secondElabList;
        QLineEdit periodDisplay;

        QLineEdit elab1Parameter;
        QLineEdit elab2Parameter;

        QVBoxLayout mainLayout;
        QHBoxLayout varLayout;
        QHBoxLayout dateLayout;
        QHBoxLayout periodLayout;
        QHBoxLayout displayLayout;
        QHBoxLayout genericPeriodLayout;

        QHBoxLayout elaborationLayout;
        QHBoxLayout readParamLayout;
        QHBoxLayout secondElabLayout;

        QList<QString> climateDbElab;
        QComboBox climateDbElabList;

        QComboBox climateDbClimaList;

    public:
        DialogAnomaly();
        void build(QSettings *settings);
        void AnomalyDisplayPeriod(const QString value);
        void AnomalyCheckYears();
        void AnomalyListElaboration(const QString value);
        void AnomalyListSecondElab(const QString value);
        void AnomalyActiveSecondParameter(const QString value);
        void AnomalyReadParameter(int state);
        void AnomalyReadReferenceState(int state);

        void AnomalySetVariableElab(const QString &value);
        QString AnomalyGetPeriodTypeList() const;
        void AnomalySetPeriodTypeList(QString period);
        bool AnomalyGetReadReferenceState();
        int AnomalyGetYearStart() const;
        int AnomalyGetYearLast() const;
        void AnomalySetYearStart(QString year);
        void AnomalySetYearLast(QString year);
        QDate AnomalyGetGenericPeriodStart() const;
        void AnomalySetGenericPeriodStart(QDate genericStart);
        QDate AnomalyGetGenericPeriodEnd() const;
        void AnomalySetGenericPeriodEnd(QDate genericEnd);
        int AnomalyGetNyears() const;
        void AnomalySetNyears(QString nYears);
        QDate AnomalyGetCurrentDay() const;
        void AnomalySetCurrentDay(QDate date);
        QString AnomalyGetElaboration() const;
        bool AnomalySetElaboration(QString elab);
        QString AnomalyGetSecondElaboration() const;
        bool AnomalySetSecondElaboration(QString elab);
        QString AnomalyGetParam1() const;
        void AnomalySetParam1(QString param);
        void AnomalySetParam1ReadOnly(bool visible);
        QString AnomalyGetParam2() const;
        void AnomalySetParam2(QString param);
        bool AnomalyReadParamIsChecked() const;
        void AnomalySetReadParamIsChecked(bool set);
        QString AnomalyGetClimateDbElab() const;
        QString AnomalyGetClimateDb() const;
        bool AnomalySetClimateDb(QString clima);
        void AnomalySetClimateDbElab(QString elab);
        void AnomalySetAllEnable(bool set);
        void AnomalyFillClimateDbList(QComboBox *dbList);
        bool AnomalyCheckValidData();
};


#endif // DIALOGANOMALY_H
