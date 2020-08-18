#include "download.h"

#include <QtNetwork>
#include <QDebug>


const QByteArray Download::_authorization = QString("Basic " + QString("ugo:Ul1ss&").toLocal8Bit().toBase64()).toLocal8Bit();

Download::Download(QString dbName, QObject* parent) : QObject(parent)
{
    _dbMeteo = new DbArkimet(dbName);
}

Download::~Download()
{
    qDebug() << "download obj destruction";
    delete _dbMeteo;
}

DbArkimet* Download::getDbArkimet()
{
    return _dbMeteo;
}

bool Download::getPointProperties(QStringList datasetList)
{

    bool result = true;
    QEventLoop loop;

    QNetworkAccessManager* manager = new QNetworkAccessManager(this);
    connect(manager, SIGNAL(finished(QNetworkReply*)), &loop, SLOT(quit()));

    _datasetsList = datasetList;

    QNetworkRequest request;
    request.setUrl(QUrl("http://meteozen.metarpa/simcstations/api/v1/stations"));
    request.setRawHeader("Authorization", _authorization);

    // GET
    QNetworkReply* reply = manager->get(request);

    loop.exec();

    if (reply->error() != QNetworkReply::NoError)
    {
            qDebug() << "Network Error: " << reply->error();
            result = false;
    }
    else
    {
        QString data = (QString) reply->readAll();

        QJsonParseError *error = new QJsonParseError();
        QJsonDocument doc = QJsonDocument::fromJson(data.toUtf8(), error);

        qDebug() << "err: " << error->errorString() << " -> " << error->offset;

        // check validity of the document
        if(!doc.isNull() && doc.isArray())
        {
            QJsonArray jsonArr = doc.array();

            for(int index = 0; index < jsonArr.size(); ++index)
            {
                QJsonObject obj = jsonArr[index].toObject();

                QJsonValue jsonDataset = obj.value("network");

                if (jsonDataset.isUndefined())
                    qDebug() << "jsonDataset: key id does not exist";
                else if (!jsonDataset.isString())
                    qDebug() << "jsonDataset: value is not string";
                else
                    foreach(QString item, _datasetsList)
                        if (jsonDataset == item)
                        {
                            this->downloadMetadata(obj);
                        }
            }
        }
         else
        {
            qDebug() << "Invalid JSON...\n" << endl;
            result = false;
        }
    }

    delete reply;
    delete manager;
    return result;
}


void Download::downloadMetadata(QJsonObject obj)
{
    Crit3DMeteoPoint* pointProp = new Crit3DMeteoPoint();

    QJsonValue jsonId = obj.value("id");

    if (jsonId.isNull())
    {
          qDebug() << "Id is empty" << endl;
          return;
    }

    int idInt = jsonId.toInt();
    pointProp->id = std::to_string(idInt);

    QJsonValue jsonName = obj.value("name");
    if (jsonName.isNull())
          qDebug() << "name is null" << endl;
    pointProp->name = jsonName.toString().toStdString();

    QJsonValue jsonNetwork = obj.value("network");
    pointProp->dataset = jsonNetwork.toString().toStdString();

    QJsonValue jsonGeometry = obj.value("geometry").toObject().value("coordinates");
    QJsonValue jsonLon = jsonGeometry.toArray()[0];
    if (jsonLon.isNull() || jsonLon.toInt() < -180 || jsonLon.toInt() > 180)
        qDebug() << "invalid Longitude" << endl;
    pointProp->longitude = jsonLon.toDouble();

    QJsonValue jsonLat = jsonGeometry.toArray()[1];
    if (jsonLat.isNull() || jsonLat.toInt() < -90 || jsonLat.toInt() > 90)
        qDebug() << "invalid Latitude" << endl;
    pointProp->latitude = jsonLat.toDouble();

    QJsonValue jsonLatInt = obj.value("lat");
    if (jsonLatInt.isNull())
        jsonLatInt = NODATA;
    pointProp->latInt = jsonLatInt.toInt();

    QJsonValue jsonLonInt = obj.value("lon");
    if (jsonLonInt.isNull())
        jsonLonInt = NODATA;
    pointProp->lonInt = jsonLonInt.toInt();

    QJsonValue jsonAltitude = obj.value("height");
    pointProp->point.z = jsonAltitude.toDouble();

    QJsonValue jsonState = obj.value("country").toObject().value("name");
    pointProp->state = jsonState.toString().toStdString();

    if (obj.value("region").isNull())
        pointProp->region = "";
    else
    {
        QJsonValue jsonRegion = obj.value("region").toObject().value("name");
        pointProp->region = jsonRegion.toString().toStdString();
    }

    if (obj.value("province").isNull())
        pointProp->province = "";
    else
    {
        QJsonValue jsonProvince = obj.value("province").toObject().value("name");
        pointProp->province = jsonProvince.toString().toStdString();
    }

    if (obj.value("municipality").isNull())
        pointProp->municipality = "";
    else
    {
        QJsonValue jsonMunicipality = obj.value("municipality").toObject().value("name");
        pointProp->municipality = jsonMunicipality.toString().toStdString();
    }

    double utmx, utmy;
    int utmZone = 32; // dove far inserire la utmZone? c'è funzione che data lat,lon restituisce utm zone?
    gis::latLonToUtmForceZone(utmZone, pointProp->latitude, pointProp->longitude, &utmx, &utmy);
    pointProp->point.utm.x = utmx;
    pointProp->point.utm.y = utmy;

    _dbMeteo->writePointProperties(pointProp);
}


bool Download::downloadDailyData(QDate startDate, QDate endDate, QString dataset, QStringList stations, QList<int> variables, bool prec0024)
{
    QString area, product, refTime;
    QDate myDate;
    QStringList fields;

    // variable properties
    QList<VariablesList> variableList = _dbMeteo->getVariableProperties(variables);

    // attenzione: il reference time dei giornalieri è a fine giornata (ore 00 di day+1)
    refTime = QString("reftime:>%1,<=%2").arg(startDate.toString("yyyy-MM-dd")).arg(endDate.addDays(1).toString("yyyy-MM-dd"));

    product = QString(";product: VM2,%1").arg(variables[0]);

    for (int i = 1; i < variables.size(); i++)
    {
        product = product % QString(" or VM2,%1").arg(variables[i]);
    }

    QEventLoop loop;

    int maxStationSize = 100;
    int j = 0;
    QUrl url;
    QNetworkRequest request;
    bool downloadOk = false;
    int countStation = 0;

    while (countStation < stations.size())
    {
        if (j == 0)
        {
            area = QString(";area: VM2,%1").arg(stations[countStation]);
            j = j+1;
            countStation = countStation+1;
        }
        while (countStation < stations.size() && j < maxStationSize)
        {
            area = area % QString(" or VM2,%1").arg(stations[countStation]);
            countStation = countStation+1;
            j = j+1;
        }

        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        connect(manager, SIGNAL(finished(QNetworkReply*)), &loop, SLOT(quit()));

        url = QUrl(QString("%1/query?query=%2%3%4&style=postprocess")
                   .arg(_dbMeteo->getDatasetURL(dataset)).arg(refTime).arg(area).arg(product));

        request.setUrl(url);
        request.setRawHeader("Authorization", _authorization);

        //std::cout << "URL: " << url.toString().toStdString(); //debug

        // GET
        QNetworkReply* reply = manager->get(request);
        downloadOk = true;
        loop.exec();

        if (reply->error() != QNetworkReply::NoError)
        {
            qDebug( "Network Error" );
            downloadOk = false;
        }
        else
        {
            _dbMeteo->createTmpTableDaily();
            bool isFirstData = true;
            QString dateStr, idPoint, flag;
            int idArkimet, idVar;
            double value;

            for (QString line = QString(reply->readLine()); !(line.isNull() || line.isEmpty());  line = QString(reply->readLine()))
            {
                fields = line.split(",");

                // warning: ref date arkimet: hour 00 of day+1
                dateStr = fields[0];
                myDate = QDate::fromString(dateStr.left(8), "yyyyMMdd");
                myDate = myDate.addDays(-1);
                dateStr = myDate.toString("yyyy-MM-dd");

                idPoint = fields[1];
                flag = fields[6];

                if (idPoint != "" && flag.left(1) != "1" && flag.left(3) != "054")
                {
                    idArkimet = fields[2].toInt();

                    if (idArkimet == PREC_ID)
                        if ((prec0024 && fields[0].mid(8,2) != "00") || (!prec0024 && fields[0].mid(8,2) != "08"))
                            continue;

                    value = fields[3].toDouble();

                    // conversion from average daily radiation to integral radiation
                    if (idArkimet == RAD_ID)
                    {
                        value *= DAY_SECONDS / 1000000.0;
                    }

                    // variable
                    int i = 0;
                    while (i < variableList.size()
                           && variableList[i].arkId() != idArkimet) i++;

                    if (i < variableList.size())
                    {
                        idVar = variableList[i].id();
                        _dbMeteo->appendQueryDaily(dateStr, idPoint, QString::number(idVar), QString::number(value), isFirstData);
                        isFirstData = false;
                    }

                }
            }

            downloadOk = _dbMeteo->saveDailyData(startDate, endDate);

            delete reply;
            delete manager;
        }

        j = 0; //reset block stations counter

    } // end while

    _dbMeteo->deleteTmpTableDaily();
    return downloadOk;
}


bool Download::downloadHourlyData(QDate startDate, QDate endDate, QString dataset, QStringList stations, QList<int> variables)
{
    // create station tables
    _dbMeteo->initStationsHourlyTables(startDate, endDate, stations);

    QList<VariablesList> variableList = _dbMeteo->getVariableProperties(variables);

    QString product = QString(";product: VM2,%1").arg(variables[0]);

    for (int i = 1; i < variables.size(); i++)
    {
        product = product % QString(" or VM2,%1").arg(variables[i]);
    }

    // start from 01:00
    QDateTime startTime((QDateTime(startDate)));
    startTime.setTimeSpec(Qt::UTC);
    startTime = startTime.addSecs(3600);

    QDateTime endTime((QDateTime(endDate)));
    endTime.setTimeSpec(Qt::UTC);
    endTime = endTime.addSecs(3600 * 24);

    // reftime
    QString refTime = QString("reftime:>=%1,<=%2").arg(startTime.toString("yyyy-MM-dd hh:mm")).arg(endTime.toString("yyyy-MM-dd hh:mm"));

    QEventLoop loop;

    int maxStationSize = 100;
    int j = 0;
    QString area;
    QUrl url;
    QNetworkRequest request;
    int countStation = 0;

    while (countStation < stations.size())
    {
        if (j == 0)
        {
            area = QString(";area: VM2,%1").arg(stations[countStation]);
            j = j+1;
            countStation = countStation+1;
        }
        while (countStation < stations.size() && j < maxStationSize)
        {
            area = area % QString(" or VM2,%1").arg(stations[countStation]);
            countStation = countStation+1;
            j = j+1;
        }
        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        connect(manager, SIGNAL(finished(QNetworkReply*)), &loop, SLOT(quit()));

        url = QUrl(QString("%1/query?query=%2%3%4&style=postprocess").arg(_dbMeteo->getDatasetURL(dataset)).arg(refTime).arg(area).arg(product));
        request.setUrl(url);
        request.setRawHeader("Authorization", _authorization);

        //qDebug() << url.toString();

        QNetworkReply* reply = manager->get(request);  // GET
        loop.exec();

        if (reply->error() != QNetworkReply::NoError)
        {
                qDebug( "Network Error" );
                delete reply;
                delete manager;
                return false;
        }
        else
        {
            _dbMeteo->queryString = "";

            QString line, dateTime, idPoint, flag, varName;
            QString idVariable, value, frequency;
            QStringList fields;
            int i, idVarArkimet;

            _dbMeteo->createTmpTableHourly();
            bool isVarOk, isFirstData = true;

            for (line = QString(reply->readLine()); !(line.isNull() || line.isEmpty());  line = QString(reply->readLine()))
            {
                fields = line.split(",");
                dateTime = QString("%1-%2-%3 %4:%5:00").arg(fields[0].left(4))
                                                           .arg(fields[0].mid(4, 2))
                                                           .arg(fields[0].mid(6, 2))
                                                           .arg(fields[0].mid(8, 2))
                                                           .arg(fields[0].mid(10, 2));
                // point
                if (fields[1] != "")
                {
                    idPoint = fields[1];

                    // variable
                    isVarOk = false;
                    idVarArkimet = fields[2].toInt();

                    for (i = 0; i < variableList.size(); i++)
                    {
                        if (variableList[i].arkId() == idVarArkimet)
                        {
                            idVariable = QString::number(variableList[i].id());
                            frequency = QString::number(variableList[i].frequency());
                            varName = variableList[i].varName();
                            isVarOk = true;
                        }
                    }

                    // value
                    if (isVarOk && fields[3] != "")
                    {
                        value = fields[3];

                        // flag
                        flag = fields[6];
                        if (flag.left(1) != "1" && flag.left(3) != "054")
                        {
                            _dbMeteo->appendQueryHourly(dateTime, idPoint, idVariable, value, isFirstData);
                            isFirstData = false;
                        }
                    }
                }
            }

            if (_dbMeteo->queryString != "")
            {
               _dbMeteo->saveHourlyData();
            }
        }

        delete reply;
        delete manager;

        j = 0; //reset block stations counter
    }

    _dbMeteo->deleteTmpTableHourly();
    return true;
}


