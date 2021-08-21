#include "importPropertiesCSV.h"
#include <QFile>
#include <QTextStream>


ImportPropertiesCSV::ImportPropertiesCSV(QString csvFileName)
    :csvFileName(csvFileName)
{

}


bool ImportPropertiesCSV::parserCSV(QString *error)
{
    if (csvFileName == "")
    {
        *error = "Missing CSV file.";
        return false;
    }

    QFile myFile(csvFileName);
    if (!myFile.open(QIODevice::ReadOnly))
    {
        *error = "Open XML failed: " + csvFileName + "\n " + myFile.errorString();
        return (false);
    }

    QTextStream myStream (&myFile);
    QList<QString> line;
    if (myStream.atEnd())
    {
        *error += "\nFile is void.";
        myFile.close();
        return false;
    }
    else
    {
        header = myStream.readLine().split(',');
    }
    while(!myStream.atEnd())
    {
        line = myStream.readLine().split(',');

        // skip void lines
        if (line.length() <= 2) continue;
        data.append(line);
    }

    myFile.close();
    return true;
}


QList<QString> ImportPropertiesCSV::getHeader() const
{
    return header;
}


QList<QList<QString> > ImportPropertiesCSV::getData() const
{
    return data;
}
