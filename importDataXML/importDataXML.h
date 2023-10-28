#ifndef IMPORTDATAXML_H
#define IMPORTDATAXML_H

#ifndef QDOM_H
    #include <QDomElement>
#endif

#include <QString>
#include <QList>
#include <QDate>
#include <QVariant>
#include "fieldXML.h"
#include "variableXML.h"
#include "dbMeteoPointsHandler.h"
#include "dbMeteoGrid.h"

enum formatType{ XMLFORMATFIXED, XMLFORMATDELIMITED};

class ImportDataXML
{
public:
    ImportDataXML(bool isGrid, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler, QString xmlFileName);
    bool parseXMLFile(QDomDocument* xmlDoc, QString *error);
    bool parserXML(QString *error);
    bool importDataMain(QString fileName, QString &error);
    QDateTime parseXMLDateTime(QString text);
    bool importXMLDataFixed(QString &error);
    bool importXMLDataDelimited(QString &error);
    QString parseXMLPointCode(QString text);
    QDate parseXMLDate(QString text);
    QVariant parseXMLFixedValue(QString text, int nReplication, FieldXML myField);

private:
    bool isGrid;
    Crit3DMeteoPointsDbHandler* meteoPointsDbHandler;
    Crit3DMeteoGridDbHandler* meteoGridDbHandler;
    QString xmlFileName;
    bool format_isSinglePoint;
    formatType format_type;
    int format_headerRow;
    float format_missingValue;
    QString format_delimiter;
    QString format_decimalSeparator;
    QString fileName_path;
    QList<QString> fileName_pragaName;
    QList<QString> fileName_fixedText;
    QList<int> fileName_nrChar;
    FieldXML time;
    FieldXML pointCode;
    FieldXML variableCode;
    QList<VariableXML> variable;
    QString dataFileName;
    int numVarFields;

    bool checkPointCodeFromFileName(QString& myPointCode, QString& errorStr);
};

#endif // IMPORTDATAXML_H
