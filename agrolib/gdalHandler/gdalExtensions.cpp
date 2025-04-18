/*
 * GDAL Extensions
 * based on Rasterix class GDALDriverHelper:
 * https://github.com/mogasw/rasterix/
*/

#include <cstring>

#include "gdalExtensions.h"
#include <gdal_priv.h>


QList<QString> getGdalExtensions(GdalFileType type, GdalFileIO io)
{
    QList<QString> extensionsList;

    const char* gdalType;
    if (type == GdalFileType::raster)
    {
        gdalType = GDAL_DCAP_RASTER;
    }
    else if(type == GdalFileType::vector)
    {
        gdalType = GDAL_DCAP_VECTOR;
    }
    else
        return extensionsList;

    int count = GetGDALDriverManager()->GetDriverCount();

    for (int i=0; i < count; i++)
    {
        GDALDriver *driver = GetGDALDriverManager()->GetDriver(i);
        if (driver)
        {
            if (std::strncmp(driver->GetDescription(), "Memory", 32) != 0)
            {
                if (driver->GetMetadataItem(gdalType) != nullptr)
                {
                    bool canRead = (driver->GetMetadataItem(GDAL_DCAP_OPEN) != nullptr);
                    bool canWrite = (driver->GetMetadataItem(GDAL_DCAP_CREATE) != nullptr) &&
                                    QString(driver->GetMetadataItem(GDAL_DMD_CREATIONDATATYPES)).startsWith("Byte");

                    QString entry = QString("%1 (*.%2)").arg(QString(driver->GetMetadataItem(GDAL_DMD_LONGNAME)),
                                                             QString(driver->GetMetadataItem(GDAL_DMD_EXTENSION)));

                    switch (io)
                    {
                        case GdalFileIO::readWrite:
                        {
                            if (canRead && canWrite)
                                extensionsList.append(entry);
                        }   break;

                        case GdalFileIO::read:
                        {
                            if (canRead)
                                extensionsList.append(entry);
                        }   break;

                        case GdalFileIO::write:
                        {
                            if (canWrite)
                                extensionsList.append(entry);
                        }   break;
                    }
                }
            }
        }
    }

    return extensionsList;
}


QList<QString> getGdalRasterReadExtension()
{
    return getGdalExtensions(GdalFileType::raster, GdalFileIO::read);
}

QList<QString> getGdalRasterWriteExtension()
{
    return getGdalExtensions(GdalFileType::raster, GdalFileIO::write);
}

QList<QString> getGdalVectorReadExtension()
{
    return getGdalExtensions(GdalFileType::vector, GdalFileIO::read);
}

QList<QString> getGdalVectorWriteExtension()
{
    return getGdalExtensions(GdalFileType::vector, GdalFileIO::write);
}


