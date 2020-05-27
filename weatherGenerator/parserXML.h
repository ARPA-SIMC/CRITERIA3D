#ifndef PARSERXML_H
#define PARSERXML_H

    #include <vector>
    #include <QString>
    #include <QStringList>
    #include <QDomElement>

    struct TXMLPoint
    {
        QString name;
        QString code;
        float latitude;
        float longitude;
        QString info;
    };

    struct TXMLClimateField
    {
        int yearFrom;
        int yearTo;
    };

    struct TXMLValuesList
    {
        QString type;
        QString attribute;
        QStringList value;
    };


    struct TXMLSeasonalAnomaly
    {
        TXMLPoint point;
        TXMLClimateField climatePeriod;
        int modelNumber;
        QStringList modelName;
        QStringList modelMember;
        int repetitions;
        int anomalyYear;
        QString anomalySeason;
        std::vector<TXMLValuesList> forecast;
    };

    void initializeSeasonalAnomaly(TXMLSeasonalAnomaly *XMLAnomaly);

    bool parseXMLFile(QString xmlFileName, QDomDocument* xmlDoc);

    bool parseXMLSeasonal(QString xmlFileName, TXMLSeasonalAnomaly* XMLAnomaly);


#endif // PARSERXML_H
