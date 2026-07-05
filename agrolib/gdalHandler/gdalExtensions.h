#ifndef GDALEXTENSIONS_H
#define GDALEXTENSIONS_H

    #include <QList>

    enum class GdalFileType : int
    {
        raster = 0,
        vector = 1
    };
    enum class GdalFileIO : int
    {
        read = 0,
        write = 1,
        readWrite = 2
    };

    QList<QString> getGdalExtensions(GdalFileType type, GdalFileIO io);

    QList<QString> getGdalRasterReadExtension();
    QList<QString> getGdalRasterWriteExtension();
    QList<QString> getGdalVectorReadExtension();
    QList<QString> getGdalVectorWriteExtension();


#endif // GDALEXTENSIONS_H
