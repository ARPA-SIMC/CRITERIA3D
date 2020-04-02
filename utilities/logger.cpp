#include "logger.h"
#include <iostream>

#include "QDir"
#include <QTextStream>

Logger::Logger()
{
    logFileName = "";
}

bool Logger::setLog(QString path, QString fileName)
{
    if (!QDir(path + "log").exists())
         QDir().mkdir(path + "log");

    m_showDate = true;
    if (!fileName.isEmpty())
    {
        file = new QFile;
        QString myDate = QDateTime().currentDateTime().toString("yyyy-MM-dd hh.mm");
        QString completefileName = fileName + "_" + myDate + ".txt";
        logFileName = path + "log/" + completefileName;
        std::cout << "SWB PROCESSOR - log file created:\n" << logFileName.toStdString() << std::endl;
        file->setFileName(logFileName);
        return file->open(QIODevice::Append | QIODevice::Text);
    }
    else
    {
        return false;
    }
}

void Logger::writeInfo(const QString &value)
{

    QString text = value;
    if (m_showDate)
    {
        text = QDateTime::currentDateTime().toString("yyyy-MM-dd hh.mm") + text;
    }
    if (logFileName != "")
    {
        QTextStream out(file);
        out.setCodec("UTF-8");
        out << text;
    }
    std::cout << text.toStdString() << std::endl;

}

void Logger::writeError(const QString &value)
{

    QString text = "----ERROR!----\n" + value;
    if (m_showDate)
    {
        text = QDateTime::currentDateTime().toString("yyyy-MM-dd hh.mm") + text;
    }
    if (logFileName != "")
    {
        QTextStream out(file);
        out.setCodec("UTF-8");
        out << text;
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
    file->close();
}
