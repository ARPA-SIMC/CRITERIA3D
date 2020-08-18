#ifndef CRITERIAAGGREGATIONVARIABLE_H
#define CRITERIAAGGREGATIONVARIABLE_H

    #include <QString>
    #include <QStringList>

    #define REQUIREDAGGREGATIONINFO 3

    class CriteriaAggregationVariable
    {
    public:
        QStringList outputVarName;
        QStringList inputField;
        QStringList computation;

        CriteriaAggregationVariable();
        bool parserAggregationVariable(QString fileName, QString &error);
    };

#endif // CRITERIAAGGREGATIONVARIABLE_H
