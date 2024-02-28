#include "fieldXML.h"
#include "commonConstants.h"

FieldXML::FieldXML()
{
    type = "";
    format = "%s";
    attribute = "";
    position = NODATA;
    firstChar = NODATA;
    nrChar = NODATA;
    alignment = "";
    prefix = "";
}

QString FieldXML::getType() const
{
    return type;
}

void FieldXML::setType(const QString &value)
{
    type = value;
}

QString FieldXML::getFormat() const
{
    return format;
}

void FieldXML::setFormat(const QString &value)
{
    format = value;
}

QString FieldXML::getAttribute() const
{
    return attribute;
}

void FieldXML::setAttribute(const QString &value)
{
    attribute = value;
}

int FieldXML::getFirstChar() const
{
    return firstChar;
}

void FieldXML::setFirstChar(int value)
{
    firstChar = value;
}

int FieldXML::getNrChar() const
{
    return nrChar;
}

void FieldXML::setNrChar(int value)
{
    nrChar = value;
}

QString FieldXML::getAlignment() const
{
    return alignment;
}

void FieldXML::setAlignment(const QString &value)
{
    alignment = value;
}

QString FieldXML::getPrefix() const
{
    return prefix;
}

void FieldXML::setPrefix(const QString &value)
{
    prefix = value;
}

int FieldXML::getPosition() const
{
    return position;
}

void FieldXML::setPosition(int value)
{
    position = value;
}
