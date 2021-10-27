#include "meteoPointsManagment.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>

QList<QString> readPointList(QString fileName)
{
    QFile pointList(fileName);
    QList<QString> points;
    if (pointList.open(QFile::ReadOnly | QFile::Text))
    {
      QTextStream sIn(&pointList);
      while (!sIn.atEnd())
      {
          QString line = sIn.readLine();
          if (points.contains(line) == 0)
          {
            points << line;
          }
      }
    }
    else
    {
      qDebug() << "error opening point list file\n";
    }
    return points;

}
