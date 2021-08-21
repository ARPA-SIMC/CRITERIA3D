#ifndef CRIT3DCLIMATE_H
#define CRIT3DCLIMATE_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef ELABORATIONSETTINGS_H
        #include "elaborationSettings.h"
    #endif
    #ifndef CRIT3DCLIMATELIST_H
        #include "crit3dClimateList.h"
    #endif

    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif
    #ifndef QDATETIME_H
        #include <QDate>
    #endif

    class Crit3DClimate
    {

    public:
        Crit3DClimate();
        ~Crit3DClimate();

        void resetParam();

        void resetCurrentValues();

        void copyParam(Crit3DClimate* clima);

        QSqlDatabase db() const;
        void setDb(const QSqlDatabase &db);

        QString climateElab() const;
        void setClimateElab(const QString &climateElab);

        int yearStart() const;
        void setYearStart(int yearStart);

        int yearEnd() const;
        void setYearEnd(int yearEnd);

        meteoVariable variable() const;
        void setVariable(const meteoVariable &variable);

        QDate genericPeriodDateStart() const;
        void setGenericPeriodDateStart(const QDate &genericPeriodDateStart);

        QDate genericPeriodDateEnd() const;
        void setGenericPeriodDateEnd(const QDate &genericPeriodDateEnd);

        int nYears() const;
        void setNYears(int nYears);

        QString elab1() const;
        void setElab1(const QString &elab1);

        float param1() const;
        void setParam1(float param1);

        bool param1IsClimate() const;
        void setParam1IsClimate(bool param1IsClimate);

        QString param1ClimateField() const;
        void setParam1ClimateField(const QString &param1ClimateField);

        QString elab2() const;
        void setElab2(const QString &elab2);

        float param2() const;
        void setParam2(float param2);

        period periodType() const;
        void setPeriodType(const period &periodType);

        QString periodStr() const;
        void setPeriodStr(const QString &periodStr);

        Crit3DElaborationSettings *getElabSettings() const;
        void setElabSettings(Crit3DElaborationSettings *value);

        meteoVariable getCurrentVar() const;
        void setCurrentVar(const meteoVariable &currentVar);

        QString getCurrentElab1() const;
        void setCurrentElab1(const QString &currentElab1);

        int getCurrentYearStart() const;
        void setCurrentYearStart(int currentYearStart);

        int getCurrentYearEnd() const;
        void setCurrentYearEnd(int currentYearEnd);

        period getCurrentPeriodType() const;
        void setCurrentPeriodType(const period &currentPeriodType);

        Crit3DClimateList *getListElab() const;
        void setListElab(Crit3DClimateList *value);
        void resetListElab();

        int getParam1ClimateIndex() const;
        void setParam1ClimateIndex(int param1ClimateIndex);

        bool getIsClimateAnomalyFromDb() const;
        void setIsClimateAnomalyFromDb(bool isClimateFromDb);

    private:
        QSqlDatabase _db;
        QString _climateElab;
        int _yearStart;
        int _yearEnd;
        period _periodType;
        meteoVariable _variable;
        QString _periodStr;
        QDate _genericPeriodDateStart;
        QDate _genericPeriodDateEnd;
        int _nYears;
        QString _elab1;
        float _param1;
        bool _param1IsClimate;
        QString _param1ClimateField;
        int _param1ClimateIndex;
        bool _isClimateAnomalyFromDb;
        QString _elab2;
        float _param2;
        Crit3DElaborationSettings *elabSettings;

        meteoVariable _currentVar;
        period _currentPeriodType;
        QString _currentElab1;
        int _currentYearStart;
        int _currentYearEnd;

        Crit3DClimateList *listElab;
    };


#endif // CRIT3DCLIMATE_H
