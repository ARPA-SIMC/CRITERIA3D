#include "utilities.h"
#include "commonConstants.h"
#include "crit3dDate.h"
#include "math.h"
#include "qjsonobject.h"

#include <QJsonDocument>
#include <QJsonArray>
#include <QVariant>
#include <QSqlDriver>
#include <QSqlRecord>
#include <QSqlQuery>
#include <QDir>
#include <QDirIterator>
#include <QTextStream>


QList<QString> getFields(QSqlDatabase* db_, QString tableName)
{
    QSqlDriver* driver_ = db_->driver();
    QSqlRecord record_ = driver_->record(tableName);
    QList<QString> fieldList;
    for (int i=0; i < record_.count(); i++)
        fieldList.append(record_.fieldName(i));

    return fieldList;
}


QList<QString> getFields(const QSqlQuery &query)
{
    QSqlRecord record = query.record();
    QList<QString> fieldList;
    for (int i=0; i < record.count(); i++)
        fieldList.append(record.fieldName(i));

    return fieldList;
}


QList<QString> getFieldsUpperCase(const QSqlQuery& query)
{
    QSqlRecord record = query.record();
    QList<QString> fieldList;
    for (int i=0; i < record.count(); i++)
        fieldList.append(record.fieldName(i).toUpper());

    return fieldList;
}


bool fieldExists(const QSqlQuery &query, const QString fieldName)
{
    QList<QString> fieldList = getFieldsUpperCase(query);
    return fieldList.contains(fieldName.toUpper());
}


// return boolean (false if recordset is not valid)
bool getValue(QVariant myRs)
{
    if (! myRs.isValid() || myRs.isNull()) return false;

    if (myRs == "" || myRs == "NULL") return false;

    return myRs.toBool();
}


bool getValue(QVariant myRs, int* myValue)
{
    *myValue = NODATA;

    if (! myRs.isValid() || myRs.isNull()) return false;
    if (myRs == "" || myRs == "NULL" || myRs == "nan") return false;

    bool isOk;
    *myValue = myRs.toInt(&isOk);

    if (! isOk)
    {
        *myValue = NODATA;
        return false;
    }

    return true;
}


bool getValue(QVariant myRs, float* myValue)
{
    *myValue = NODATA;

    if (! myRs.isValid() || myRs.isNull()) return false;
    if (myRs == "" || myRs == "NULL" || myRs == "nan") return false;

    bool isOk;
    *myValue = myRs.toFloat(&isOk);

    if (! isOk)
    {
        *myValue = NODATA;
        return false;
    }

    return true;
}


bool getValue(QVariant myRs, double* myValue)
{
    *myValue = NODATA;

    if (! myRs.isValid() || myRs.isNull()) return false;
    if (myRs == "" || myRs == "NULL" || myRs == "nan") return false;

    bool isOk;
    *myValue = myRs.toDouble(&isOk);

    if (! isOk)
    {
        *myValue = NODATA;
        return false;
    }

    return true;
}


bool getValue(QVariant myRs, QDate* myValue)
{
    if (myRs.isNull())
        return false;
    else
    {
        if (myRs == "")
             return false;
        else
            *myValue = myRs.toDate();
    }

    return true;
}

bool getValue(QVariant myRs, QDateTime* myValue)
{
    if (myRs.isNull())
        return false;
    else
    {
        if (myRs == "")
             return false;
        else
            *myValue = myRs.toDateTime();
    }

    return true;
}


bool getValue(QVariant myRs, QString* myValue)
{
    *myValue = "";
    if (! myRs.isValid() || myRs.isNull()) return false;
    if (myRs == "NULL") return false;

    *myValue = myRs.toString();
    return true;
}


Crit3DDate getCrit3DDate(const QDate& d)
{
    Crit3DDate myDate = Crit3DDate(d.day(), d.month(), d.year());
    return myDate;
}


Crit3DTime getCrit3DTime(const QDateTime& t)
{
    Crit3DTime myTime;

    myTime.date.day = t.date().day();
    myTime.date.month = t.date().month();
    myTime.date.year = t.date().year();
    myTime.time = t.time().hour()*3600 + t.time().minute()*60 + t.time().second();

    return myTime;
}


Crit3DTime getCrit3DTime(const QDate& t, int hour)
{
    Crit3DTime myTime;

    myTime.date.day = t.day();
    myTime.date.month = t.month();
    myTime.date.year = t.year();

    if (hour >= 24)
    {
        int nrDays = int(floor(hour / 24));
        for (int i = 1; i <= nrDays; i++)
            ++myTime.date;
        hour -= (nrDays * 24);
    }
    myTime.time = hour * 3600;

    return myTime;
}


QDate getQDate(const Crit3DDate& d)
{
    QDate myDate = QDate(d.year, d.month, d.day);
    return myDate;
}


QDateTime getQDateTime(const Crit3DTime& t)
{
    QDate myDate = QDate(t.date.year, t.date.month, t.date.day);

    QDateTime myDateTime;
    myDateTime.setTimeSpec(Qt::UTC);
    myDateTime.setDate(myDate);
    myDateTime.setTime(QTime(0,0,0,0));
    return myDateTime.addSecs(t.time);
}


QString getFileName(const QString &fileNameComplete)
{
    QString c;
    QString fileName = "";
    for (int i = fileNameComplete.length()-1; i >= 0; i--)
    {
        c = fileNameComplete.mid(i,1);
        if ((c != "\\") && (c != "/"))
        {
            fileName = c + fileName;
        }
        else
        {
            return fileName;
        }
    }

    return fileName;
}


QString getFilePath(const QString &fileNameComplete)
{
    QString fileName = getFileName(fileNameComplete);
    QString filePath = fileNameComplete.left(fileNameComplete.length() - fileName.length());
    return filePath;
}


int decadeFromDate(QDate date)
{
    int day = date.day();
    int decade;
    if ( day <= 10)
    {
        decade = 1;
    }
    else if ( day <= 20)
    {
        decade = 2;
    }
    else
    {
        decade = 3;
    }
    decade = decade + (3 * (date.month() - 1));
    return decade;
}


void intervalDecade(int decade, int year, int* dayStart, int* dayEnd, int* month)
{
    int decMonth;

    *month = ((decade - 1) / 3) + 1;
    if ( (decade % 3) == 0)
    {
        decMonth = 3;
    }
    else
    {
        decMonth = decade % 3;
    }

    if (decMonth == 1)
    {
        *dayStart = 1;
        *dayEnd = 10;
    }
    else if (decMonth == 2)
    {
        *dayStart = 11;
        *dayEnd = 20;
    }
    else
    {
        *dayStart = 21;
        QDate temp(year, *month, 1);
        *dayEnd = temp.daysInMonth();
    }

}

int getSeasonFromDate(QDate date)
{
    int month = date.month();

    switch (month) {
    case 3: case 4: case 5:
        return 1;
    case 6: case 7: case 8:
        return 2;
    case 9: case 10: case 11:
        return 3;
    case 12: case 1: case 2:
        return 4;
    default:
        return NODATA;
    }
}

int getSeasonFromString(QString season)
{

    if (season == "MAM")
    {
        return 1;
    }
    else if (season == "JJA")
    {
        return 2;
    }
    else if (season == "SON")
    {
        return 3;
    }
    else if (season == "DJF")
    {
        return 4;
    }
    else
        return NODATA;
}


QString getStringSeasonFromDate(QDate date)
{

    int month = date.month();
    switch (month) {
    case 3: case 4: case 5:
        return "MAM";
    case 6: case 7: case 8:
        return "JJA";
    case 9: case 10: case 11:
        return "SON";
    case 12: case 1: case 2:
        return "DJF";
    default:
        return "";
    }
}


bool getPeriodDates(QString periodSelected, int year, QDate myDate, QDate* startDate, QDate* endDate)
{

   startDate->setDate(year, myDate.month(), myDate.day());
   endDate->setDate(year, myDate.month(), myDate.day());

   if (periodSelected == "Daily")
   {
        return true;
   }
   if (periodSelected == "Decadal")
   {
       int decade = decadeFromDate(myDate);
       int dayStart;
       int dayEnd;
       int month;
       intervalDecade(decade, myDate.year(), &dayStart, &dayEnd, &month);
       startDate->setDate(startDate->year(), startDate->month(), dayStart);
       endDate->setDate(endDate->year(), endDate->month(), dayEnd);
   }
   else if (periodSelected == "Monthly")
   {
       startDate->setDate(startDate->year(), startDate->month(), 1);
       endDate->setDate(endDate->year(), endDate->month(), endDate->daysInMonth());
   }
   else if (periodSelected == "Seasonal")
   {
       int mySeason = getSeasonFromDate(myDate);
       if ( (myDate.month() == 12) || (myDate.month() == 1) || (myDate.month() == 2))
       {
           startDate->setDate( (startDate->year() - 1), 12, 1);
           QDate temp(endDate->year(), 2, 1);
           endDate->setDate(endDate->year(), 2, temp.daysInMonth());
       }
       else
       {
           startDate->setDate(startDate->year(), mySeason*3, 1);
           QDate temp(endDate->year(), mySeason*3+2, 1);
           endDate->setDate(endDate->year(), mySeason*3+2, temp.daysInMonth());
       }

   }
   else if (periodSelected == "Annual")
   {
       startDate->setDate(startDate->year(), 1, 1);
       endDate->setDate(endDate->year(), 12, 31);
   }
   else
   {
       return false;
   }
   return true;

}

std::vector <float> StringListToFloat(QList<QString> myList)
{
    std::vector <float> myVector;
    myVector.resize(unsigned(myList.size()));
    for (unsigned i=0; i < unsigned(myList.size()); i++)
        myVector[i] = myList[int(i)].toFloat();

    return myVector;
}

std::vector <double> StringListToDouble(QList<QString> myList)
{
    std::vector <double> myVector;
    myVector.resize(unsigned(myList.size()));
    for (unsigned i=0; i < unsigned(myList.size()); i++)
        myVector[i] = myList[int(i)].toFloat();

    return myVector;
}

std::vector<int> StringListToInt(QList<QString> myList)
{
    std::vector <int> myVector;
    myVector.resize(unsigned(myList.size()));
    for (unsigned i=0; i < unsigned(myList.size()); i++)
        myVector[i] = myList[int(i)].toInt();

    return myVector;
}

QStringList FloatVectorToStringList(std::vector <float> myVector)
{
    QList<QString> myList;
    for (unsigned i=0; i < unsigned(myVector.size()); i++)
        myList.push_back(QString::number(double(myVector[i])));

    return myList;
}

QStringList DoubleVectorToStringList(std::vector <double> myVector)
{
    QList<QString> myList;
    for (unsigned i=0; i < unsigned(myVector.size()); i++)
        myList.push_back(QString::number(double(myVector[i])));

    return myList;
}

QStringList IntVectorToStringList(std::vector <int> myVector)
{
    QList<QString> myList;
    for (unsigned i=0; i < unsigned(myVector.size()); i++)
        myList.push_back(QString::number(int(myVector[i])));

    return myList;
}

bool removeDirectory(QString myPath)
{
    QDir myDir(myPath);
    myDir.setNameFilters(QList<QString>() << "*.*");
    myDir.setFilter(QDir::Files);

    // remove all files
    foreach(QString myFile, myDir.entryList())
    {
        myDir.remove(myFile);
    }

    return myDir.rmdir(myPath);
}


bool searchDocPath(QString* docPath)
{
    *docPath = "";

    QString myPath = QDir::currentPath();
    QString myRoot = QDir::rootPath();
    // only for win: application can run on a different drive (i.e. D:\)
    QString winRoot = myPath.left(3);

    bool isFound = false;
    while (! isFound)
    {
        if (QDir(myPath + "/DOC").exists())
        {
            isFound = true;
            break;
        }

        if (QDir::cleanPath(myPath) == myRoot || QDir::cleanPath(myPath) == winRoot)
            break;

        myPath = QFileInfo(myPath).dir().absolutePath();
    }
    if (! isFound) return false;

    *docPath = QDir::cleanPath(myPath) + "/DOC/";
    return true;
}


bool searchDataPath(QString* dataPath)
{
    *dataPath = "";

    QString myPath = QDir::currentPath();
    QString myRoot = QDir::rootPath();
    // only for win: application can run on a different drive (i.e. D:\)
    QString winRoot = myPath.left(3);

    bool isFound = false;
    while (! isFound)
    {
        if (QDir(myPath + "/DATA").exists())
        {
            isFound = true;
            break;
        }

        if (QDir::cleanPath(myPath) == myRoot || QDir::cleanPath(myPath) == winRoot)
            break;

        myPath = QFileInfo(myPath).dir().absolutePath();
    }
    if (! isFound) return false;

    *dataPath = QDir::cleanPath(myPath) + "/DATA/";
    return true;
}

void clearDir( const QString path )
{
    QDir dir( path );

    dir.setFilter( QDir::NoDotAndDotDot | QDir::Files );
    foreach( QString dirItem, dir.entryList() )
        dir.remove( dirItem );

    dir.setFilter( QDir::NoDotAndDotDot | QDir::Dirs );
    foreach( QString dirItem, dir.entryList() )
    {
        QDir subDir( dir.absoluteFilePath( dirItem ) );
        subDir.removeRecursively();
    }
}


QList<QString> readListSingleColumn(QString fileName, QString& error)
{
    QFile listFile(fileName);
    QList<QString> myList;
    if (listFile.open(QFile::ReadOnly | QFile::Text))
    {
        QTextStream sIn(&listFile);
        while (!sIn.atEnd())
        {
            QString newValue = sIn.readLine();
            if (newValue != "" && myList.contains(newValue) == 0)
            {
                myList << newValue;
            }
        }
    }
    else
    {
        error = "Wrong file format (must be a text file): " + fileName;
    }

    return myList;
}


QList<QString> removeList(const QList<QString> &list, QList<QString> &toDelete)
{
    QList<QString> newList = list;

    QList<QString>::iterator i;
    for (i = toDelete.begin(); i != toDelete.end(); ++i)
    {
        newList.removeAll(*i);
    }

    return newList;
}


// remove files from targetPath, containing targetStr in the name and older than nrDays
void removeOldFiles(const QString &targetPath, const QString &targetStr, int nrDays)
{
    // iterate through the directory using the QDirIterator
    QDirIterator it(targetPath);

    while (it.hasNext())
    {
        QString filename = it.next();
        QFileInfo file(filename);

        if (file.isDir())
            continue;

        if (file.fileName().contains(targetStr, Qt::CaseInsensitive))
        {
            if (file.fileTime(QFileDevice::FileModificationTime) < QDateTime::currentDateTime().addDays(-nrDays))
            {
                QFile myFile(filename);
                myFile.remove();
            }
        }
    }
}


bool parseCSV(const QString &csvFileName, QList<QString> &csvFields, QList<QList<QString>> &csvData, QString &errorString)
{
    if (csvFileName.isEmpty() || ! QFile(csvFileName).exists() || ! QFileInfo(csvFileName).isFile())
    {
        errorString = "Missing file: " + csvFileName;
        return false;
    }

    QFile myFile(csvFileName);
    if (! myFile.open(QIODevice::ReadOnly))
    {
        errorString = "Open failed: " + csvFileName + "\n " + myFile.errorString();
        return false;
    }

    QTextStream myStream (&myFile);
    if (myStream.atEnd())
    {
        errorString = "File is void";
        myFile.close();
        return false;
    }
    else
    {
        csvFields = myStream.readLine().split(',');
    }

    csvData.clear();
    while(! myStream.atEnd())
    {
        QList<QString> line = myStream.readLine().split(',');

        // skip void lines
        if (line.size() <= 1) continue;
        csvData.append(line);
    }

    myFile.close();
    return true;
}

bool writeJson(const QString & ancestor, const std::vector <QString> &fieldNames, const std::vector <QString> dataType, const std::vector <std::vector <QString>> &values, const QString & jsonFilename)
{
    QJsonObject content;
    QJsonArray records;
    QJsonObject recordObject;

    bool isFloat = false;

    for (int i=0; i < int(values.size()); i++)
    {
        if (values[i].size() != fieldNames.size() || values[i].size() != dataType.size()) return false;

        recordObject.empty();
        for (int j=0; j < int(values[i].size()); j++)
        {
            if (dataType[j] == "float")
                recordObject.insert(fieldNames[j], values[i][j].toFloat(&isFloat));
            else
                recordObject.insert(fieldNames[j], values[i][j]);
        }

        records.push_back(recordObject);
    }

    content.insert(ancestor, records);

    QJsonDocument doc(content);
    QByteArray bytes = doc.toJson(QJsonDocument::Indented);
    QFile file(jsonFilename);
    if(file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate ))
    {
        QTextStream iStream( &file );
        iStream << bytes;
        file.close();
        return true;
    }
    else
        return false;
}
