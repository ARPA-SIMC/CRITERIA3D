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
        QList<QList<QString> > getData() const;

    private:
        QString csvFileName;
        QList<QString> header;
        QList<QList<QString>> data;
    };

#endif // IMPORTPROPERTIESCSV_H
