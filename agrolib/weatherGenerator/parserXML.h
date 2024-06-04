#ifndef PARSERXML_H
#define PARSERXML_H

    #include <vector>
    #include <QString>
    #include <QStringList>
    #include <QDomElement>

    struct TXMLScenarioPoint
    {
        QString name;
        QString code;
        float latitude;
        float longitude;
        float height;
    };
    struct TXMLScenarioModels
    {
        QString type;
        QStringList value;
        int number;
    };
    struct TXMLScenarioClimateField
    {
        int yearFrom;
        int yearTo;
    };
    struct TXMLScenarioType
    {
        QString type;
        int yearFrom;
        int yearTo;
    };
    struct TXMLScenarioValuesList
    {
        QString type;
        QString attribute;
        QStringList value;
    };
    struct TXMLScenarioPeriod
    {
        QString type;
        std::vector<TXMLScenarioValuesList> seasonalScenarios;
    };

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


    class XMLSeasonalAnomaly
    {
    public:
        XMLSeasonalAnomaly();

        void initialize();
        void printInfo();

        TXMLPoint point;
        std::vector<TXMLValuesList> forecast;
        TXMLClimateField climatePeriod;
        int modelNumber;
        QStringList modelName;
        QStringList modelMember;
        int repetitions;
        int anomalyYear;
        QString anomalySeason;
    };

    class XMLScenarioAnomaly
    {
    public:
        XMLScenarioAnomaly();

        void initialize();
        void printInfo();

        TXMLScenarioPoint point;
        TXMLScenarioModels models;
        TXMLScenarioClimateField climatePeriod;
        TXMLScenarioType scenario;
        TXMLScenarioPeriod period[4]; // four season
        int repetitions;
        int anomalyYear;
    };

    bool parseXMLFile(const QString &xmlFileName, QDomDocument &xmlDoc);

    bool parseXMLSeasonal(const QString &xmlFileName, XMLSeasonalAnomaly &XMLAnomaly);

    bool parseXMLScenario(const QString &xmlFileName, XMLScenarioAnomaly &XMLAnomaly);


#endif // PARSERXML_H
