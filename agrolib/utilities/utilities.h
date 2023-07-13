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

    QList<QString> getFields(QSqlDatabase* db_, QString tableName);
    QList<QString> getFields(const QSqlQuery& query);
    QList<QString> getFieldsUpperCase(const QSqlQuery &query);

    bool getValue(QVariant myRs);
    bool getValue(QVariant myRs, int* myValue);
    bool getValue(QVariant myRs, float* myValue);
    bool getValue(QVariant myRs, double* myValue);
    bool getValue(QVariant myRs, QDate* myValue);
    bool getValue(QVariant myRs, QDateTime* myValue);
    bool getValue(QVariant myRs, QString* myValue);

    QString getFilePath(const QString &fileNameComplete);
    QString getFileName(const QString &fileNameComplete);

    std::vector <float> StringListToFloat(QList<QString> myList);
    QStringList FloatVectorToStringList(std::vector <float> myVector);
    QList<QString> readListSingleColumn(QString fileName, QString& error);

    bool removeDirectory(QString myPath);
    bool searchDocPath(QString* docPath);
    bool searchDataPath(QString* dataPath);

    void removeOldFiles(const QString &targetPath, const QString &targetStr, int nrDays);

    void clearDir( const QString path );
    QList<QString> removeList(QList<QString> list, QList<QString> toDelete);


#endif // UTILITIES_H
