#ifndef IMPORTDATAXML_H
#define IMPORTDATAXML_H

#ifndef QDOM_H
    #include <QDomElement>
#endif

#include <QString>
#include <QList>
#include "fieldXML.h"
#include "variableXML.h"

class ImportDataXML
{
public:
    ImportDataXML(bool isGrid, QString xmlFileName);
    bool parseXMLFile(QDomDocument* xmlDoc, QString *error);
    bool parserXML(QString *error);
    bool importData(QString fileName, QString *error);
private:
    bool isGrid;
    QString xmlFileName;
    bool format_isSinglePoint;
    bool format_isFixed;
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
};

#endif // IMPORTDATAXML_H
