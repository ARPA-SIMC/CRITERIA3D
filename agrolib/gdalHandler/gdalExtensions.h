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

    QStringList getGdalExtensions(GdalFileType type, GdalFileIO io);

    QStringList getGdalRasterReadExtension();
    QStringList getGdalRasterWriteExtension();
    QStringList getGdalVectorReadExtension();
    QStringList getGdalVectorWriteExtension();


#endif // GDALEXTENSIONS_H
