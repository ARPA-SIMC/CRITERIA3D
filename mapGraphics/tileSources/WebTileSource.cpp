#include "WebTileSource.h"
#include "guts/MapGraphicsNetwork.h"

#include <cmath>
#include <QPainter>
#include <QStringBuilder>
#include <QtDebug>
#include <QNetworkReply>

#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    #include <QtCore5Compat/QRegExp>
#endif


const qreal PI = 3.14159265358979323846;
const qreal deg2rad = PI / 180.0;
const qreal rad2deg = 180.0 / PI;

WebTileSource::WebTileSource(WebTileType tileType) :
    MapTileSource(), _tileType(tileType)
{
    this->setCacheMode(MapTileSource::DiskAndMemCaching);
}

WebTileSource::~WebTileSource()
{
    qDebug() << this << this->name() << "Destructing";
}

QPointF WebTileSource::ll2qgs(const QPointF &ll, quint8 zoomLevel) const
{
    const qreal tilesOnOneEdge = pow(2.0,zoomLevel);
    const quint16 tileSize = this->tileSize();
    qreal x = (ll.x()+180) * (tilesOnOneEdge*tileSize)/360; // coord to pixel!
    qreal y = (1-(log(tan(PI/4+(ll.y()*deg2rad)/2)) /PI)) /2  * (tilesOnOneEdge*tileSize);

    return QPoint(int(x), int(y));
}

QPointF WebTileSource::qgs2ll(const QPointF &qgs, quint8 zoomLevel) const
{
    const qreal tilesOnOneEdge = pow(2.0,zoomLevel);
    const quint16 tileSize = this->tileSize();
    qreal longitude = (qgs.x()*(360/(tilesOnOneEdge*tileSize)))-180;
    qreal latitude = rad2deg*(atan(sinh((1-qgs.y()*(2/(tilesOnOneEdge*tileSize)))*PI)));

    return QPointF(longitude, latitude);
}

quint64 WebTileSource::tilesOnZoomLevel(quint8 zoomLevel) const
{
    return quint64(pow(4.0, zoomLevel));
}

quint16 WebTileSource::tileSize() const
{
    return 256;
}

quint8 WebTileSource::minZoomLevel(QPointF ll)
{
    Q_UNUSED(ll)
    return 0;
}

quint8 WebTileSource::maxZoomLevel(QPointF ll)
{
    Q_UNUSED(ll)
    return 18;
}

QString WebTileSource::name() const
{
    switch(_tileType)
    {
    case OPEN_STREET_MAP:
        return "OpenStreetMap standard tiles";

    case GOOGLE_MAP:
        return "Google Map Tiles";

    case GOOGLE_Satellite:
        return "Google Satellite Tiles";

    case GOOGLE_Hybrid_Satellite:
        return "Google Hybrid Satellite Map Tiles";

    case GOOGLE_Terrain:
        return "Google Terrain Tiles";

    case ESRI_WorldImagery:
        return "ESRI - World Imagery tiles";

    case STAMEN_Terrain:
        return "Stamen Terrain tiles";   
    }

    return "Unknown tiles";
}

QString WebTileSource::tileFileExtension() const
{
    if (_tileType == OPEN_STREET_MAP ||
        _tileType == ESRI_WorldImagery ||
        _tileType == STAMEN_Terrain ||
        _tileType == GOOGLE_MAP)
        return "png";
    else
        return "jpg";
}

//protected
void WebTileSource::fetchTile(quint32 x, quint32 y, quint8 z)
{
    MapGraphicsNetwork * network = MapGraphicsNetwork::getInstance();

    QString host;
    QString url;

    //Figure out which server to request from based on our desired tile type
    if (_tileType == OPEN_STREET_MAP)
    {
        host = "http://c.tile.openstreetmap.org";
        url = "/%1/%2/%3.png";
    }
    else if (_tileType == GOOGLE_MAP)
    {
        host = "http://mt.google.com/vt/lyrs=m&";
        url = "x=%2&y=%3&z=%1";
    }
    else if (_tileType == GOOGLE_Satellite)
    {
        host = "http://mt.google.com/vt/lyrs=s&";
        url = "x=%2&y=%3&z=%1";
    }
    else if (_tileType == GOOGLE_Hybrid_Satellite)
    {
        host = "http://mt.google.com/vt/lyrs=y&";
        url = "x=%2&y=%3&z=%1";
    }
    else if (_tileType == GOOGLE_Terrain)
    {
        host = "http://mt.google.com/vt/lyrs=p&";
        url = "x=%2&y=%3&z=%1";
    }
    else if (_tileType == STAMEN_Terrain)
    {
        host = "http://tile.stamen.com";
        url = "/terrain/%1/%2/%3.png";
    }
    else if (_tileType == ESRI_WorldImagery)
    {
        host = "http://server.arcgisonline.com";
        url = "/arcgis/rest/services/World_Imagery/MapServer/tile/%1/%3/%2.png";
    }
    else
    {
        // default
        host = "http://b.tile.openstreetmap.org";
        url = "/%1/%2/%3.png";
    }


    //Use the unique cacheID to see if this tile has already been requested
    const QString cacheID = this->createCacheID(x,y,z);
    if (_pendingRequests.contains(cacheID))
        return;
    _pendingRequests.insert(cacheID);

    //Build the request
    const QString fetchURL = url.arg(QString::number(z),
                                     QString::number(x),
                                     QString::number(y));
    QNetworkRequest request(QUrl(host + fetchURL));

    //Send the request and setupd a signal to ensure we're notified when it finishes
    QNetworkReply * reply = network->get(request);
    _pendingReplies.insert(reply,cacheID);

    connect(reply,
            SIGNAL(finished()),
            this,
            SLOT(handleNetworkRequestFinished()));
}

//private slot
void WebTileSource::handleNetworkRequestFinished()
{
    QObject * sender = QObject::sender();
    QNetworkReply * reply = qobject_cast<QNetworkReply *>(sender);

    if (reply == nullptr)
    {
        qWarning() << "QNetworkReply cast failure";
        return;
    }

    /*
      We can do this here and use reply later in the function because the reply
      won't be deleted until execution returns to the event loop.
    */
    reply->deleteLater();

    if (!_pendingReplies.contains(reply))
    {
        qWarning() << "Unknown QNetworkReply";
        return;
    }

    //get the cacheID
    const QString cacheID = _pendingReplies.take(reply);
    _pendingRequests.remove(cacheID);

    //If there was a network error, ignore the reply
    if (reply->error() != QNetworkReply::NoError)
    {
        qDebug() << "Network Error:" << reply->errorString();
        return;
    }

    //Convert the cacheID back into x,y,z tile coordinates
    quint32 x,y,z;
    if (!MapTileSource::cacheID2xyz(cacheID,&x,&y,&z))
    {
        qWarning() << "Failed to convert cacheID" << cacheID << "back to xyz";
        return;
    }

    QByteArray bytes = reply->readAll();
    QImage * image = new QImage();

    if (!image->loadFromData(bytes))
    {
        delete image;
        qWarning() << "Failed to make QImage from network bytes";
        return;
    }

    //Figure out how long the tile should be cached
    QDateTime expireTime = QDateTime::currentDateTimeUtc().addSecs(86400*30);       // one month

    if (reply->hasRawHeader("Cache-Control"))
    {
        //We support the max-age directive only for now
        const QByteArray cacheControl = reply->rawHeader("Cache-Control");
        QRegExp maxAgeFinder("max-age=(\\d+)");
        if (maxAgeFinder.indexIn(cacheControl) != -1)
        {
            bool ok = false;
            const qint64 delta = maxAgeFinder.cap(1).toULongLong(&ok);

            if (ok)
                expireTime = QDateTime::currentDateTimeUtc().addSecs(delta);
        }
    }

    //Notify client of tile retrieval
    this->prepareNewlyReceivedTile(x,y,z, image, expireTime);
}
