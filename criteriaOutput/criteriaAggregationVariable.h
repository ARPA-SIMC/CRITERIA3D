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

        CriteriaAggregationVariable();
        bool parserAggregationVariable(QString fileName, QString &error);
    };

#endif // CRITERIAAGGREGATIONVARIABLE_H
