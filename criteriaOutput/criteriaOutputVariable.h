#ifndef CRITERIAOUTPUTVARIABLE_H
#define CRITERIAOUTPUTVARIABLE_H

    #include <QString>
    #include <QList>

    #define CSVREQUIREDINFO 8

    class CriteriaOutputVariable
    {
    public:
        QList<QString> outputVarNameList;
        QList<QString> varNameList;
        QList<QString> computationList;
        QList<int> referenceDay;
        QList<QString> nrDays;
        QList<QString> climateComputation;
        QList<int> param1;
        QList<int> param2;

        CriteriaOutputVariable();
        bool parserOutputVariable(QString fileName, QString &error);
    };

#endif // CRITERIAOUTPUTVARIABLE_H
