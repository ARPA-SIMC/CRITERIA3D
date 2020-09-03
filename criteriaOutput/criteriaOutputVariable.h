#ifndef CRITERIAOUTPUTVARIABLE_H
#define CRITERIAOUTPUTVARIABLE_H

    #include <QString>
    #include <QStringList>

    #define CSVREQUIREDINFO 8

    class CriteriaOutputVariable
    {
    public:
        QStringList outputVarName;
        QStringList varName;
        QList<int> referenceDay;
        QStringList computation;
        QStringList nrDays;
        QStringList climateComputation;
        QList<int> param1;
        QList<int> param2;

        CriteriaOutputVariable();
        bool parserOutputVariable(QString fileName, QString &error);
    };

#endif // CRITERIAOUTPUTVARIABLE_H
