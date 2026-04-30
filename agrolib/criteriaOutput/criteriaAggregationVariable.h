#ifndef CRITERIAAGGREGATIONVARIABLE_H
#define CRITERIAAGGREGATIONVARIABLE_H

    #include <QString>
    #include <QList>

    #define REQUIREDAGGREGATIONINFO 3

    class CriteriaAggregationVariable
    {
    public:
        QList<QString> outputVarName;
        QList<QString> inputFieldName;
        QList<QString> aggregationType;

        CriteriaAggregationVariable() { }
        bool parserAggregationVariable(const QString fileName, QString &errorStr);
    };

#endif // CRITERIAAGGREGATIONVARIABLE_H
