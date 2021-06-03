#ifndef IMPORTDATAXML_H
#define IMPORTDATAXML_H

#ifndef QDOM_H
    #include <QDomElement>
#endif

#include <QString>
#include <QList>
#include "fieldXML.h"

class ImportDataXML
{
public:
    ImportDataXML(bool isGrid, QString xmlFileName);
    bool parseXMLFile(QDomDocument* xmlDoc, QString *error);
    bool parserXML(QString *error);
private:
    bool isGrid;
    QString xmlFileName;
    bool isSinglePoint;
    bool format_isFixed;
    int headerRow;
    float missingValue;
    QString delimiter;
    QString decimalSeparator;
    QString fileName_path;
    QList<QString> fileName_pragaName;
    QList<QString> fileName_fixedText;
    QList<int> fileName_nrChar;
    FieldXML time;
    FieldXML pointCode;
    FieldXML variableCode;
};

#endif // IMPORTDATAXML_H
