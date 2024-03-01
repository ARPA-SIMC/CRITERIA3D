#ifndef VARIABLEXML_H
#define VARIABLEXML_H

#include <QString>
#include "fieldXML.h"

class VariableXML
{
    public:
        VariableXML();
        FieldXML varField;
        FieldXML flagField;
        QString flagAccepted;
        int nReplication;
};

#endif // VARIABLEXML_H
