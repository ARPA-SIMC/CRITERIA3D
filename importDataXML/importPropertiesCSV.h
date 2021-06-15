#ifndef IMPORTPROPERTIESCSV_H
#define IMPORTPROPERTIESCSV_H

#include <QString>
#include <QList>

class ImportPropertiesCSV
{
public:
    ImportPropertiesCSV(QString csvFileName);
    bool parserCSV(QString *error);
    QList<QString> getHeader() const;

private:
    QString csvFileName;
    QList<QString> header;
};

#endif // IMPORTPROPERTIESCSV_H
