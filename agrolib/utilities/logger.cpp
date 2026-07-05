#include "logger.h"

#include <iostream>
#include <QDir>
#include <QTextStream>
#include <QDateTime>

Logger::Logger()
{
    logFileName = "";
    file = nullptr;
}

bool Logger::setLog(QString path, QString fileName, bool addDateTime)
{
    if (! QDir(path + "log").exists())
    {
         QDir().mkdir(path + "log");
    }

    m_showDate = true;
    if (!fileName.isEmpty())
    {
        file = new QFile;
        if (addDateTime)
        {
            QString myDate = QDateTime().currentDateTime().toString("yyyy-MM-dd hh.mm");
            logFileName = path + "log/" + fileName + " " + myDate + ".txt";
        }
        else
        {
            logFileName = path + "log/" + fileName + ".txt";
        }

        logFileName = QDir().cleanPath(logFileName);
        file->setFileName(logFileName);
        std::cout << "Log file created: " << logFileName.toStdString() << std::endl;
        return file->open(QIODevice::WriteOnly | QIODevice::Text);
    }
    else
    {
        return false;
    }
}


void Logger::writeInfo(const QString &infoStr)
{
    QString text = infoStr;
    if (m_showDate)
    {
        text = QDateTime::currentDateTime().toString("yyyy-MM-dd hh.mm") + " " + text;
    }
    if (logFileName != "")
    {
        QTextStream out(file);
        //out.setCodec("UTF-8");
        out << text + "\n";
    }
    std::cout << text.toStdString() << std::endl;
}


void Logger::writeError(const QString &errorStr)
{
    QString text = " ----ERROR!---- \n" + errorStr;
    if (m_showDate)
    {
        text = QDateTime::currentDateTime().toString("yyyy-MM-dd hh.mm") + text;
    }
    if (logFileName != "")
    {
        QTextStream out(file);
        //out.setCodec("UTF-8");
        out << text + "\n";
    }
    std::cout << text.toStdString() << std::endl;
}


void Logger::setShowDateTime(bool value)
{
    m_showDate = value;
}


Logger::~Logger()
{
    if (file != nullptr)
    {
        file->close();
    }
}
