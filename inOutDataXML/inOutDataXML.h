#ifndef INOUTDATAXML_H
#define INOUTDATAXML_H

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

class InOutDataXML
{
public:
    InOutDataXML(bool isGrid, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler, QString xmlFileName);
    bool parseXMLFile(QDomDocument* xmlDoc, QString *error);
    bool parserXML(QString *error);
    bool importDataMain(QString fileName, QString &error);
    QDateTime parseXMLDateTime(QString text);
    bool importXMLDataFixed(QString &error);
    bool importXMLDataDelimited(QString &error);
    QString parseXMLPointCode(QString text);
    QDate parseXMLDate(QString text);
    QVariant parseXMLFixedValue(QString text, int nReplication, FieldXML myField);
    bool checkPointCodeFromFileName(QString& myPointCode, QString& errorStr);
    QString parseXMLFilename(QString code);
    QString getVariableExport();
    int getPointCodeFirstChar();
    int getPointCodeNrChar();
    int getVariableCodeFirstChar();
    QString getVariableCodeAttribute();
    int getVariableCodeNrChar();
    QString getVariableAlign();
    int getVariableFirstChar();
    int getVariableNrChar();
    QString getVariableFormat();
    int getTimeFirstChar();
    QString getTimeFormat();
    QString getTimeType();
    QString getVariableFlagAccepted();
    int getVariableFlagFirstChar();
    float getFormatMissingValue();

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
    QString fileName_pragaName;
    QList<QString> fileName_fixedPrefix;
    QList<QString> fileName_fixedSuffix;
    int fileName_nrChar;
    FieldXML time;
    FieldXML pointCode;
    FieldXML variableCode;
    QList<VariableXML> variable;
    QString dataFileName;
};

#endif // INOUTDATAXML_H
