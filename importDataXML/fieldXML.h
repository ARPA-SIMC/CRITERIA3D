#ifndef FIELDXML_H
#define FIELDXML_H

#include <QString>

class FieldXML
{
public:
    FieldXML();
    QString getType() const;
    void setType(const QString &value);

    QString getFormat() const;
    void setFormat(const QString &value);

    QString getAttribute() const;
    void setAttribute(const QString &value);

    QString getField() const;
    void setField(const QString &value);

    int getFirstChar() const;
    void setFirstChar(int value);

    int getNrChar() const;
    void setNrChar(int value);

    QString getAlignment() const;
    void setAlignment(const QString &value);

    QString getPrefix() const;
    void setPrefix(const QString &value);

private:
    QString type;
    QString format;
    QString attribute;
    QString field;
    int firstChar;
    int nrChar;
    QString alignment;
    QString prefix;
};

#endif // FIELDXML_H
