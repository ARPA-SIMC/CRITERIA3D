#ifndef LOGGER_H
#define LOGGER_H

    #include <QFile>
    #include <QTextStream>
    #include <QDateTime>

    class Logger
    {
    public:
        Logger();
        ~Logger();

        bool setLog(QString path, QString fileName);
        void setShowDateTime(bool value);
        void writeInfo(const QString &value);
        void writeError(const QString &value);

    private:
         QFile *file;
         QString logFileName;
         bool m_showDate;

    };

#endif // LOGGER_H
