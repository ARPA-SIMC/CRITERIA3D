/*!
* \brief functions on date, string, files and recordset
*/

#ifndef UTILITIES_H
#define UTILITIES_H

    #include <vector>
    #include <QVariant>
    #include <QSqlDatabase>
    #include <QDateTime>

    class Crit3DDate;
    class Crit3DTime;
    class QVariant;

    Crit3DDate getCrit3DDate(const QDate &myDate);
    Crit3DTime getCrit3DTime(const QDateTime &myTime);
    Crit3DTime getCrit3DTime(const QDate& t, int hour);

    QDate getQDate(const Crit3DDate &myDate);
    QDateTime getQDateTime(const Crit3DTime &myCrit3DTime);
    int decadeFromDate(QDate date);
    void intervalDecade(int decade, int year, int* dayStart, int* dayEnd, int* month);
    int getSeasonFromDate(QDate date);
    int getSeasonFromString(QString season);
    QString getStringSeasonFromDate(QDate date);
    bool getPeriodDates(QString periodSelected, int year, QDate myDate, QDate* startDate, QDate* endDate);

    QList<QString> getFields(QSqlDatabase* dbPointer, QString tableName);
    QList<QString> getFields(const QSqlQuery& query);
    QList<QString> getFieldsUpperCase(const QSqlQuery &query);
    bool fieldExists(const QSqlQuery &query, const QString fieldName);

    bool getValue(const QVariant &myRs);
    bool getValue(const QVariant &myRs, int* value);
    bool getValue(const QVariant &myRs, float* value);
    bool getValue(const QVariant &myRs, double* value);
    bool getValue(const QVariant &myRs, QString* valueStr);
    bool getValue(const QVariant &myRs, QDate* date);
    bool getValue(const QVariant &myRs, QDateTime* dateTime);
    bool getValueCrit3DTime(const QVariant &myRs, Crit3DTime *dateTime);

    QString getFilePath(const QString &fileNameComplete);
    QString getFileName(const QString &fileNameComplete);

    std::vector <float> StringListToFloat(QList<QString> myList);
    std::vector <double> StringListToDouble(QList<QString> myList);
    std::vector <int> StringListToInt(QList<QString> myList);
    QStringList FloatVectorToStringList(std::vector <float> myVector);
    QStringList DoubleVectorToStringList(std::vector <double> myVector);
    QStringList IntVectorToStringList(std::vector <int> myVector);
    QList<QString> readListSingleColumn(QString fileName, QString& error);

    bool removeDirectory(QString myPath);
    bool searchDocPath(QString &docPath);
    bool searchDataPath(QString* dataPath);

    void removeOldFiles(const QString &targetPath, const QString &targetStr, int nrDays);

    void clearDir( const QString path);
    QList<QString> removeList(const QList<QString> &list, QList<QString> &toDelete);

    bool parseCSV(const QString &csvFileName, QList<QString> &csvFields, QList<QList<QString>> &csvData, QString &errorString);

    bool writeJson(const QString &ancestor, const std::vector<QString> &fieldNames, const std::vector<QString> dataType, const std::vector<std::vector<QString> > &values, const QString &jsonFilename);

#endif // UTILITIES_H
