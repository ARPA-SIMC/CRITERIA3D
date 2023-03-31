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

    bool parseXMLFile(QString xmlFileName, QDomDocument* xmlDoc);

    bool parseXMLSeasonal(QString xmlFileName, XMLSeasonalAnomaly* XMLAnomaly);


#endif // PARSERXML_H
