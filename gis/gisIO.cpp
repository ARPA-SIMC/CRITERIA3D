#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <cctype>

#include "commonConstants.h"
#include "basicMath.h"
#include "gis.h"

using namespace std;


enum class RasterDataType
{
    UInt8,
    Int16,
    Int32,
    Float32
};


enum class ByteOrder
{
    LittleEndian,
    BigEndian
};

struct EsriBilInfo
{
    int nRows = 0;
    int nCols = 0;
    int nBands = 1;
    int nBits = 0;

    std::string pixelType;
    char byteOrder = 'I';

    double ulx = 0.0;
    double uly = 0.0;
    double xDim = 0.0;
    double yDim = 0.0;

    float noData = NODATA;
};


struct RasterFileInfo
{
    RasterDataType dataType = RasterDataType::Float32;
    ByteOrder byteOrder = ByteOrder::LittleEndian;
    std::size_t headerOffset = 0;
};


string upperCase(const string& myStr)
{
    string result = myStr;

    transform(
        result.begin(),
        result.end(),
        result.begin(),
        [](unsigned char c)
        {
            return static_cast<char>(toupper(c));
        });

    return result;
}


string lowerCase(const string& myStr)
{
    string result = myStr;

    transform(
        result.begin(),
        result.end(),
        result.begin(),
        [](unsigned char c)
        {
            return static_cast<char>(tolower(c));
        });

    return result;
}


bool isLittleEndianHost()
{
    const uint16_t value = 1;

    return *reinterpret_cast<const uint8_t*>(&value) == 1;
}


template <typename T>
T swapBytes(T value)
{
    T result;

    const uint8_t* src =
        reinterpret_cast<const uint8_t*>(&value);

    uint8_t* dst =
        reinterpret_cast<uint8_t*>(&result);

    for (size_t i = 0; i < sizeof(T); ++i)
        dst[i] = src[sizeof(T) - i - 1];

    return result;
}


template <typename T>
T convertByteOrder(T value, ByteOrder fileOrder)
{
    const bool hostLittle = isLittleEndianHost();
    const bool fileLittle =
        (fileOrder == ByteOrder::LittleEndian);

    if (hostLittle == fileLittle)
        return value;

    return swapBytes(value);
}


bool parseByteOrder(const string& value, ByteOrder& byteOrder)
{
    const string s = upperCase(trim(value));

    if (s == "0" || s == "LSBFIRST" || s == "LITTLEENDIAN")
    {
        byteOrder = ByteOrder::LittleEndian;
        return true;
    }

    if (s == "1" || s == "MSBFIRST" || s == "BIGENDIAN")
    {
        byteOrder = ByteOrder::BigEndian;
        return true;
    }

    return false;
}


bool getEnviDataType(int enviType, RasterDataType& dataType, int& nrBytes)
{
    switch (enviType)
    {
    case 1:
        dataType = RasterDataType::UInt8;
        nrBytes = 1;
        return true;

    case 2:
        dataType = RasterDataType::Int16;
        nrBytes = 2;
        return true;

    case 3:
        dataType = RasterDataType::Int32;
        nrBytes = 4;
        return true;

    case 4:
        dataType = RasterDataType::Float32;
        nrBytes = 4;
        return true;

    default:
        return false;
    }
}


bool getDataTypeFromBytes(int nrBytes, RasterDataType& dataType)
{
    switch (nrBytes)
    {
    case 1:
        dataType = RasterDataType::UInt8;
        return true;

    case 2:
        dataType = RasterDataType::Int16;
        return true;

    case 4:
        dataType = RasterDataType::Float32;
        return true;

    default:
        return false;
    }
}


bool readRasterRow(FILE* file,
                   float* destination,
                   int nCols,
                   const RasterFileInfo& fileInfo,
                   vector<uint8_t>& buffer)
{
    if (file == nullptr || destination == nullptr || nCols <= 0)
        return false;

    const size_t bytesPerValue =
        [&]()
    {
        switch (fileInfo.dataType)
        {
        case RasterDataType::UInt8:
            return size_t(1);

        case RasterDataType::Int16:
            return size_t(2);

        case RasterDataType::Int32:
            return size_t(4);

        case RasterDataType::Float32:
            return size_t(4);
        }

        return size_t(0);
    }();

    if (bytesPerValue == 0)
        return false;

    const size_t rowSize =
        static_cast<size_t>(nCols) * bytesPerValue;

    buffer.resize(rowSize);

    if (fread(buffer.data(), 1, rowSize, file) != rowSize)
        return false;

    const bool needsSwap =
        (fileInfo.byteOrder == ByteOrder::BigEndian) ==
        isLittleEndianHost();

    for (int col = 0; col < nCols; ++col)
    {
        const uint8_t* src =
            buffer.data() +
            static_cast<size_t>(col) * bytesPerValue;

        switch (fileInfo.dataType)
        {
        case RasterDataType::UInt8:
        {
            destination[col] =
                static_cast<float>(*src);

            break;
        }

        case RasterDataType::Int16:
        {
            int16_t value;

            memcpy(&value, src, sizeof(value));

            if (needsSwap)
                value = swapBytes(value);

            destination[col] =
                static_cast<float>(value);

            break;
        }

        case RasterDataType::Int32:
        {
            int32_t value;

            memcpy(&value, src, sizeof(value));

            if (needsSwap)
                value = swapBytes(value);

            destination[col] =
                static_cast<float>(value);

            break;
        }

        case RasterDataType::Float32:
        {
            float value;

            memcpy(&value, src, sizeof(value));

            if (needsSwap)
                value = swapBytes(value);

            destination[col] = value;

            break;
        }
        }
    }

    return true;
}


bool readRasterBinaryData(const string& fileName,
                          gis::Crit3DRasterGrid* rasterGrid,
                          const RasterFileInfo& fileInfo,
                          string& errorStr)
{
    errorStr.clear();

    if (rasterGrid == nullptr ||
        rasterGrid->header == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    if (rasterGrid->header->nrRows <= 0 ||
        rasterGrid->header->nrCols <= 0)
    {
        errorStr = "Invalid raster dimensions.";
        return false;
    }

    if (!rasterGrid->initializeGrid())
    {
        errorStr = "Memory error: file too big.";
        return false;
    }

    FILE* file = fopen(fileName.c_str(), "rb");

    if (file == nullptr)
    {
        errorStr =
            "Error opening raster file: " +
            fileName + "\n" +
            strerror(errno);

        rasterGrid->clear();
        return false;
    }

    if (fileInfo.headerOffset > 0)
    {
        if (fseek(file,
                  static_cast<long>(fileInfo.headerOffset),
                  SEEK_SET) != 0)
        {
            errorStr =
                "Error seeking raster data: " +
                fileName;

            fclose(file);
            rasterGrid->clear();
            return false;
        }
    }

    vector<uint8_t> rowBuffer;

    const int nRows = rasterGrid->header->nrRows;
    const int nCols = rasterGrid->header->nrCols;

    for (int row = 0; row < nRows; ++row)
    {
        if (!readRasterRow(file,
                           rasterGrid->value[row],
                           nCols,
                           fileInfo,
                           rowBuffer))
        {
            errorStr =
                "Error reading raster data at row " +
                to_string(row) + ".";

            fclose(file);
            rasterGrid->clear();
            return false;
        }
    }

    fclose(file);

    return true;
}


bool parseKeyValue(const string& line,
                   string& key,
                   string& value)
{
    key.clear();
    value.clear();

    const string cleaned = trim(line);

    if (cleaned.empty())
        return false;

    istringstream stream(cleaned);

    if (!(stream >> key))
        return false;

    if (!(stream >> value))
        return false;

    return true;
}


bool parseKeyValueByDelimiter(const string& line,
                              const string& delimiter,
                              string& key,
                              string& value)
{
    key.clear();
    value.clear();

    const size_t pos = line.find(delimiter);

    if (pos == string::npos)
        return false;

    key = trim(line.substr(0, pos));
    value = trim(line.substr(pos + delimiter.size()));

    return !key.empty() && !value.empty();
}


vector<string> splitCommaDelimited(const string& str)
{
    vector<string> result;

    string item;
    stringstream stream(str);

    while (getline(stream, item, ','))
        result.push_back(trim(item));

    return result;
}


void removeBraces(string& str)
{
    str.erase(
        remove_if(
            str.begin(),
            str.end(),
            [](unsigned char c)
            {
                return c == '{' || c == '}';
            }),
        str.end());
}


bool isValidHeaderValue(double value)
{
    return std::isfinite(value);
}


bool splitKeyValue(const string& str, string& key, string& value)
{
    return parseKeyValue(str, key, value);
}


void cleanSpaces(string& str)
{
    str.erase(
        remove_if(
            str.begin(),
            str.end(),
            [](unsigned char c)
            {
                return isspace(c);
            }),
        str.end());
}


bool splitKeyValueByDelimiter(const string& myLine,
                              const string& delimiter,
                              string& key,
                              string& value)
{
    return parseKeyValueByDelimiter(
        myLine,
        delimiter,
        key,
        value);
}



namespace gis
{


bool readEsriFloatHeader(const string& fileName, Crit3DRasterHeader* header, string& errorStr)
{
    errorStr.clear();

    if (header == nullptr)
    {
        errorStr = "Invalid raster header.";
        return false;
    }

    string fn = fileName;

    const string extension = lowerCase(
        fn.size() >= 4 ?
            fn.substr(fn.size() - 4) :
            "");

    if (extension == ".flt")
        fn.resize(fn.size() - 4);

    const string headerFileName = fn + ".hdr";

    ifstream file(headerFileName);

    if (!file.is_open())
    {
        errorStr = "Missing file: " + headerFileName;
        return false;
    }

    bool hasCols = false;
    bool hasRows = false;
    bool hasCellSize = false;
    bool hasLLX = false;
    bool hasLLY = false;
    bool hasNoData = false;

    string line, key, value;

    while (getline(file, line))
    {
        if (!parseKeyValue(line, key, value))
            continue;

        const string upKey = upperCase(key);

        if (upKey == "NCOLS")
        {
            int valueInt;

            if (!parseInt(value, valueInt) || valueInt <= 0)
            {
                errorStr = "Invalid NCOLS value.";
                return false;
            }

            header->nrCols = valueInt;
            hasCols = true;
        }
        else if (upKey == "NROWS")
        {
            int valueInt;

            if (!parseInt(value, valueInt) || valueInt <= 0)
            {
                errorStr = "Invalid NROWS value.";
                return false;
            }

            header->nrRows = valueInt;
            hasRows = true;
        }
        else if (upKey == "XLLCORNER")
        {
            double valueDouble;

            if (!parseDouble(value, valueDouble) || !isValidHeaderValue(valueDouble))
            {
                errorStr = "Invalid XLLCORNER value.";
                return false;
            }

            header->llCorner.x = valueDouble;
            hasLLX = true;
        }
        else if (upKey == "YLLCORNER")
        {
            double valueDouble;

            if (!parseDouble(value, valueDouble) || !isValidHeaderValue(valueDouble))
            {
                errorStr = "Invalid YLLCORNER value.";
                return false;
            }

            header->llCorner.y = valueDouble;
            hasLLY = true;
        }
        else if (upKey == "CELLSIZE")
        {
            double valueDouble;

            if (!parseDouble(value, valueDouble) ||
                !isValidHeaderValue(valueDouble) ||
                valueDouble <= 0.)
            {
                errorStr = "Invalid CELLSIZE value.";
                return false;
            }

            header->cellSize = valueDouble;
            header->invCellSize = 1.0 / valueDouble;
            hasCellSize = true;
        }
        else if (upKey == "NODATA_VALUE" || upKey == "NODATA")
        {
            float valueFloat;

            if (!parseFloat(value, valueFloat))
            {
                errorStr = "Invalid NODATA value.";
                return false;
            }

            header->flag = valueFloat;
            hasNoData = true;
        }
        else if (upKey == "DATATYPE")
        {
            int nrBytes;

            if (!parseInt(value, nrBytes))
            {
                errorStr = "Invalid DATATYPE value: " + value;
                return false;
            }

            header->nrBytes = nrBytes;
        }
    }

    if (!hasCols || !hasRows ||
        !hasCellSize ||
        !hasLLX || !hasLLY ||
        !hasNoData)
    {
        errorStr = "Missing keys in ESRI header file.";
        return false;
    }

    /*
     * ESRI .flt is normally Float32.
     * nrBytes is retained for compatibility with
     * Crit3DRasterHeader.
     */
    if (header->nrBytes == 0)
        header->nrBytes = 4;

    return true;
}


bool readEsriBilHeader(const std::string& fileName, gis::Crit3DRasterHeader* header,
                       EsriBilInfo& bilInfo, std::string& errorStr)
{
    errorStr.clear();

    if (header == nullptr)
    {
        errorStr = "Invalid raster header.";
        return false;
    }

    std::ifstream file(fileName + ".hdr");

    if (!file.is_open())
    {
        errorStr = "Missing file: " + fileName + ".hdr";
        return false;
    }

    std::string line;

    while (std::getline(file, line))
    {
        std::string key;
        std::string value;

        if (!splitKeyValue(line, key, value))
            continue;

        key = upperCase(key);

        try
        {
            if (key == "NROWS")
            {
                bilInfo.nRows = std::stoi(value);
            }
            else if (key == "NCOLS")
            {
                bilInfo.nCols = std::stoi(value);
            }
            else if (key == "NBANDS")
            {
                bilInfo.nBands = std::stoi(value);
            }
            else if (key == "NBITS")
            {
                bilInfo.nBits = std::stoi(value);
            }
            else if (key == "PIXELTYPE")
            {
                bilInfo.pixelType = upperCase(value);
            }
            else if (key == "BYTEORDER")
            {
                bilInfo.byteOrder = static_cast<char>(
                    std::toupper(static_cast<unsigned char>(value[0])));
            }
            else if (key == "ULXMAP")
            {
                bilInfo.ulx = std::stod(value);
            }
            else if (key == "ULYMAP")
            {
                bilInfo.uly = std::stod(value);
            }
            else if (key == "XDIM")
            {
                bilInfo.xDim = std::stod(value);
            }
            else if (key == "YDIM")
            {
                bilInfo.yDim = std::stod(value);
            }
            else if (key == "NODATA")
            {
                bilInfo.noData = std::stof(value);
            }
        }
        catch (const std::exception&)
        {
            errorStr = "Invalid value in BIL header: " + key;
            return false;
        }
    }

    file.close();

    if (bilInfo.nRows <= 0 || bilInfo.nCols <= 0)
    {
        errorStr = "Invalid number of rows or columns in BIL header.";
        return false;
    }

    if (bilInfo.nBands != 1)
    {
        errorStr = "Only single-band BIL rasters are supported.";
        return false;
    }

    if (bilInfo.nBits != 8 &&
        bilInfo.nBits != 16 &&
        bilInfo.nBits != 32)
    {
        errorStr = "Unsupported BIL bit depth: " + std::to_string(bilInfo.nBits);
        return false;
    }

    if (bilInfo.xDim <= 0.0 || bilInfo.yDim <= 0.0)
    {
        errorStr = "Invalid cell size in BIL header.";
        return false;
    }

    header->nrRows = bilInfo.nRows;
    header->nrCols = bilInfo.nCols;
    header->nrBytes = bilInfo.nBits / 8;

    header->cellSize = bilInfo.xDim;
    header->invCellSize = 1.0 / header->cellSize;

    /*
     * ESRI BIL ULXMAP/ULYMAP normally refer to the center of the upper-left pixel.
     * Crit3DRasterHeader uses the lower-left corner.
     */
    header->llCorner.x = bilInfo.ulx - 0.5 * bilInfo.xDim;

    header->llCorner.y =  bilInfo.uly - bilInfo.nRows * bilInfo.yDim
                         + 0.5 * bilInfo.yDim;

    header->flag = bilInfo.noData;

    return true;
}


bool readEsriBilProjection(const std::string& fileName, int &utmZone, std::string& errorStr)
{
    errorStr.clear();

    const std::string prjFileName = fileName + ".prj";

    std::ifstream myFile(prjFileName);

    if (!myFile.is_open())
    {
        errorStr = "Missing projection file: " + prjFileName;
        return false;
    }

    // Read entire WKT
    std::ostringstream buffer;
    buffer << myFile.rdbuf();
    myFile.close();

    std::string wkt = buffer.str();

    if (wkt.empty())
    {
        errorStr = "Empty projection file: " + prjFileName;
        return false;
    }

    // Uppercase for comparisons
    std::string upWkt = upperCase(wkt);

    // Check UTM
    if (upWkt.find("UTM") == std::string::npos)
    {
        errorStr = "Only UTM projection is allowed.";
        return false;
    }

    // Check WGS-84
    if (upWkt.find("WGS_1984") == std::string::npos &&
        upWkt.find("WGS-84") == std::string::npos &&
        upWkt.find("WGS 84") == std::string::npos &&
        upWkt.find("WGS84") == std::string::npos)
    {
        errorStr = "Only WGS-84 datum is allowed.";
        return false;
    }

    // Find UTM zone
    const std::string zoneKey = "ZONE_";

    std::size_t zonePos = upWkt.find(zoneKey);

    if (zonePos == std::string::npos)
    {
        errorStr = "UTM zone not found in projection.";
        return false;
    }

    zonePos += zoneKey.length();

    std::size_t zoneEnd = zonePos;

    while (zoneEnd < upWkt.size() &&
           std::isdigit(static_cast<unsigned char>(upWkt[zoneEnd])))
    {
        ++zoneEnd;
    }

    if (zoneEnd == zonePos)
    {
        errorStr = "Invalid UTM zone.";
        return false;
    }

    try
    {
        utmZone = std::stoi(upWkt.substr(zonePos, zoneEnd - zonePos));
    }
    catch (...)
    {
        errorStr = "Invalid UTM zone.";
        return false;
    }

    return true;
}


bool isBigEndianSystem()
{
    const uint16_t value = 0x0102;
    return reinterpret_cast<const unsigned char*>(&value)[0] == 0x01;
}

bool readRasterBilData(const std::string& fileName, Crit3DRasterGrid* rasterGrid,
                       const EsriBilInfo& bilInfo, std::string& errorStr)
{
    if (rasterGrid == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    FILE* filePointer = std::fopen(fileName.c_str(), "rb");

    if (filePointer == nullptr)
    {
        errorStr = "Error opening BIL file: " + fileName
                   + "\n" + std::strerror(errno);
        return false;
    }

    const bool fileBigEndian = (bilInfo.byteOrder == 'M');

    const bool systemBigEndian = isBigEndianSystem();

    const bool swap = fileBigEndian != systemBigEndian;

    const int nRows = bilInfo.nRows;
    const int nCols = bilInfo.nCols;

    if (bilInfo.nBits == 8)
    {
        std::vector<uint8_t> buffer(nCols);

        for (int row = 0; row < nRows; ++row)
        {
            if (std::fread(buffer.data(), sizeof(uint8_t), nCols, filePointer) !=
                static_cast<size_t>(nCols))
            {
                errorStr = "Error reading BIL raster data.";
                std::fclose(filePointer);
                return false;
            }

            for (int col = 0; col < nCols; ++col)
            {
                rasterGrid->value[row][col] = static_cast<float>(buffer[col]);
            }
        }
    }
    else if (bilInfo.nBits == 16)
    {
        std::vector<int16_t> buffer(nCols);

        for (int row = 0; row < nRows; ++row)
        {
            if (std::fread(buffer.data(), sizeof(int16_t),  nCols, filePointer) !=
                static_cast<size_t>(nCols))
            {
                errorStr = "Error reading BIL raster data.";
                std::fclose(filePointer);
                return false;
            }

            for (int col = 0; col < nCols; ++col)
            {
                int16_t value = buffer[col];

                if (swap)
                    value = swapBytes(value);

                rasterGrid->value[row][col] = static_cast<float>(value);
            }
        }
    }
    else if (bilInfo.nBits == 32)
    {
        if (bilInfo.pixelType == "FLOAT")
        {
            std::vector<float> buffer(nCols);

            for (int row = 0; row < nRows; ++row)
            {
                if (std::fread(buffer.data(), sizeof(float), nCols, filePointer) !=
                    static_cast<size_t>(nCols))
                {
                    errorStr = "Error reading BIL raster data.";
                    std::fclose(filePointer);
                    return false;
                }

                for (int col = 0; col < nCols; ++col)
                {
                    float value = buffer[col];

                    if (swap)
                        value = swapBytes(value);

                    rasterGrid->value[row][col] = value;
                }
            }
        }
        else
        {
            std::vector<int32_t> buffer(nCols);

            for (int row = 0; row < nRows; ++row)
            {
                if (std::fread(buffer.data(), sizeof(int32_t), nCols, filePointer) !=
                    static_cast<size_t>(nCols))
                {
                    errorStr = "Error reading BIL raster data.";
                    std::fclose(filePointer);
                    return false;
                }

                for (int col = 0; col < nCols; ++col)
                {
                    int32_t value = buffer[col];

                    if (swap)
                        value = swapBytes(value);

                    rasterGrid->value[row][col] = static_cast<float>(value);
                }
            }
        }
    }

    std::fclose(filePointer);

    return true;
}


bool readEsriGridBil(const std::string& fileName, Crit3DRasterGrid* rasterGrid,
                      int currentUtmZone, std::string& errorStr)
{
    errorStr.clear();

    if (rasterGrid == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    rasterGrid->clear();

    EsriBilInfo bilInfo;
    if (!readEsriBilHeader(fileName, rasterGrid->header, bilInfo, errorStr))
        return false;

    int utmZone;
    if (! readEsriBilProjection(fileName, utmZone, errorStr))
        return false;

    if (utmZone != currentUtmZone)
    {
        errorStr = "UTM zone: " + std::to_string(utmZone) + "\n" +
                   "is different from current UTM zone: " + std::to_string(currentUtmZone);
        return false;
    }

    if (!rasterGrid->initializeGrid())
    {
        errorStr = "Memory error: file too big.";
        return false;
    }

    if (!readRasterBilData(fileName + ".bil", rasterGrid, bilInfo, errorStr))
    {
        rasterGrid->clear();
        return false;
    }

    gis::updateMinMaxRasterGrid(rasterGrid);
    rasterGrid->isLoaded = true;

    return true;
}


bool readEnviHeader(const string& fileName, Crit3DRasterHeader* header,
                    int currentUtmZone, RasterFileInfo& fileInfo, string& errorStr)
{
    errorStr.clear();

    if (header == nullptr)
    {
        errorStr = "Invalid raster header.";
        return false;
    }

    string headerFileName = fileName + ".hdr";
    ifstream file(headerFileName);

    if (! file.is_open())
    {
        headerFileName = fileName + ".img.hdr";
        file.open(headerFileName);
    }

    if (! file.is_open())
    {
        errorStr = "Missing file: " + fileName + ".hdr";
        return false;
    }

    bool hasSamples = false;
    bool hasLines = false;
    bool hasMapInfo = false;
    bool hasDataType = false;
    bool hasNoData = false;
    bool hasByteOrder = false;

    string line, key, value;

    while (getline(file, line))
    {
        if (! parseKeyValueByDelimiter(line, "=", key, value))
            continue;

        cleanSpaces(key);

        const string upKey = upperCase(key);

        if (upKey == "SAMPLES")
        {
            int valueInt;

            if (! parseInt(value, valueInt) || valueInt <= 0)
            {
                errorStr = "Invalid ENVI samples value.";
                return false;
            }

            header->nrCols = valueInt;
            hasSamples = true;
        }
        else if (upKey == "LINES")
        {
            int valueInt;

            if (! parseInt(value, valueInt) || valueInt <= 0)
            {
                errorStr = "Invalid ENVI lines value.";
                return false;
            }

            header->nrRows = valueInt;
            hasLines = true;
        }
        else if (upKey == "DATATYPE")
        {
            int enviType;

            if (! parseInt(value, enviType))
            {
                errorStr = "Invalid ENVI data type: " + value;
                return false;
            }

            if (! getEnviDataType(enviType, fileInfo.dataType, header->nrBytes))
            {
                errorStr = "Unsupported ENVI data type: " + value;
                return false;
            }

            hasDataType = true;
        }
        else if (upKey == "BYTEORDER")
        {
            if (! parseByteOrder(value, fileInfo.byteOrder))
            {
                errorStr = "Unsupported ENVI byte order: " + value;
                return false;
            }

            hasByteOrder = true;
        }
        else if (upKey == "HEADEROFFSET")
        {
            int offset;
            if (! parseInt(value, offset) || offset < 0)
            {
                errorStr = "Invalid ENVI header offset: " + value;
                return false;
            }

            fileInfo.headerOffset = static_cast<size_t>(offset);
        }
        else if (upKey == "DATAIGNOREVALUE" ||
                 upKey == "NODATA")
        {
            float valueFloat;

            if (! parseFloat(value, valueFloat))
            {
                errorStr = "Invalid ENVI data ignore value.";
                return false;
            }

            header->flag = valueFloat;
            hasNoData = true;
        }
        else if (upKey == "MAPINFO")
        {
            string mapInfo = value;
            removeBraces(mapInfo);

            vector<string> info = splitCommaDelimited(mapInfo);

            if (info.size() < 10)
            {
                errorStr = "Incomplete ENVI map info.";
                return false;
            }

            for (string& item : info)
                item = trim(item);

            if (upperCase(info[0]) != "UTM")
            {
                errorStr = "Only UTM projection is allowed.";
                return false;
            }

            const string datum = upperCase(info[9]);

            if (datum != "WGS84" && datum != "WGS-84")
            {
                errorStr = "Only WGS-84 datum is allowed.";
                return false;
            }

            int utmZone;
            if (! parseInt(info[7], utmZone))
            {
                errorStr = "Invalid UTM zone: " + info[7];
                return false;
            }

            if (utmZone != currentUtmZone)
            {
                errorStr = "UTM zone: " + info[7]
                           + "\nCurrent UTM zone is " + to_string(currentUtmZone);
                return false;
            }

            double cellSizeX;
            double cellSizeY;

            if (! parseDouble(info[5], cellSizeX) ||
                ! parseDouble(info[6], cellSizeY) ||
                cellSizeX <= 0. || cellSizeY <= 0.)
            {
                errorStr = "Invalid ENVI cell size.";
                return false;
            }

            if (! isEqual(cellSizeX, cellSizeY))
            {
                errorStr = "Different cell sizes on x and y are not allowed.";
                return false;
            }

            double xTopLeft;
            double yTopLeft;

            if (! parseDouble(info[3], xTopLeft) ||
                ! parseDouble(info[4], yTopLeft))
            {
                errorStr = "Invalid ENVI map coordinates.";
                return false;
            }

            header->cellSize = cellSizeX;
            header->invCellSize = 1.0 / cellSizeX;

            header->llCorner.x = xTopLeft;
            header->llCorner.y = yTopLeft - static_cast<double>(header->nrRows) * header->cellSize;

            hasMapInfo = true;
        }
    }

    if (! hasSamples
        || ! hasLines
        || ! hasMapInfo)
    {
        errorStr = "Wrong ENVI header file: missing samples, lines or map info.";
        return false;
    }

    if (! hasDataType)
    {
        errorStr = "Wrong ENVI header file: missing data type.";
        return false;
    }

    if (! hasNoData)
    {
        errorStr = "Wrong ENVI header file: missing data ignore value.";
        return false;
    }

    /*
     * ENVI defaults to little endian when byte order
     * is not explicitly specified.
     */
    if (! hasByteOrder)
        fileInfo.byteOrder = ByteOrder::LittleEndian;

    return true;
}


bool readRasterFloatData(const string& fileName,
                         Crit3DRasterGrid* rasterGrid,
                         string& errorStr)
{
    if (rasterGrid == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    RasterFileInfo fileInfo;

    fileInfo.dataType =
        RasterDataType::Float32;

    fileInfo.byteOrder =
        ByteOrder::LittleEndian;

    fileInfo.headerOffset = 0;

    return readRasterBinaryData(
        fileName,
        rasterGrid,
        fileInfo,
        errorStr);
}


bool readRasterFloatData(const string& fileName,
                         Crit3DRasterGrid* rasterGrid,
                         const RasterFileInfo& fileInfo,
                         string& errorStr)
{
    return readRasterBinaryData(
        fileName,
        rasterGrid,
        fileInfo,
        errorStr);
}


bool writeEsriGridHeader(const string& fileName,
                         Crit3DRasterHeader* header,
                         string& errorStr)
{
    errorStr.clear();

    if (header == nullptr)
    {
        errorStr = "Invalid raster header.";
        return false;
    }

    const string outputFileName =
        fileName + ".hdr";

    ofstream file(outputFileName);

    if (!file.is_open())
    {
        errorStr =
            "Error writing file: " +
            outputFileName +
            "\n" +
            strerror(errno);

        return false;
    }

    file << "ncols         "
         << header->nrCols << '\n';

    file << "nrows         "
         << header->nrRows << '\n';

    file << "xllcorner     "
         << header->llCorner.x << '\n';

    file << "yllcorner     "
         << header->llCorner.y << '\n';

    file << "cellsize      "
         << header->cellSize << '\n';

    file << "NODATA_value  "
         << header->flag << '\n';

    /*
     * ESRI .flt is a 32-bit floating-point
     * little-endian raster.
     */
    file << "byteorder     LSBFIRST\n";

    if (!file.good())
    {
        errorStr =
            "Error writing file: " +
            outputFileName;

        return false;
    }

    return true;
}


bool writeEsriGridFlt(const string& fileName,
                      Crit3DRasterGrid* rasterGrid,
                      string& errorStr)
{
    errorStr.clear();

    if (rasterGrid == nullptr || rasterGrid->header == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    const string outputFileName = fileName + ".flt";

    FILE* file = fopen(outputFileName.c_str(), "wb");

    if (file == nullptr)
    {
        errorStr =
            "Error writing file: " +
            outputFileName +
            "\n" +
            strerror(errno);

        return false;
    }

    const int nRows = rasterGrid->header->nrRows;
    const int nCols = rasterGrid->header->nrCols;

    for (int row = 0; row < nRows; ++row)
    {
        const size_t written =
            fwrite(
                rasterGrid->value[row],
                sizeof(float),
                static_cast<size_t>(nCols),
                file);

        if (written !=
            static_cast<size_t>(nCols))
        {
            errorStr =
                "Error writing raster data at row " +
                to_string(row) +
                ".";

            fclose(file);
            return false;
        }
    }

    if (fclose(file) != 0)
    {
        errorStr =
            "Error closing file: " +
            outputFileName;

        return false;
    }

    return true;
}


bool writeEsriGrid(const string& fileName, Crit3DRasterGrid* rasterGrid, string& errorStr)
{
    if (! writeEsriGridHeader(fileName, rasterGrid->header, errorStr))
        return false;

    if (! writeEsriGridFlt(fileName, rasterGrid, errorStr))
        return false;

    return true;
}


bool readEsriGridFlt(const string& fileName, Crit3DRasterGrid* rasterGrid, string& errorStr)
{
    errorStr.clear();

    if (rasterGrid == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    rasterGrid->clear();

    string extension;
    if (fileName.size() >= 4)
    {
        std::string suffix = lowerCase(fileName.substr(fileName.size() - 4));
        if (suffix[0] == '.')
            extension = suffix;
    }

    if (! extension.empty() && extension != ".flt")
    {
        errorStr = "Invalid raster suffix.";
        return false;
    }

    if (! readEsriFloatHeader(fileName, rasterGrid->header, errorStr))
    {
        rasterGrid->clear();
        return false;
    }

    RasterFileInfo fileInfo;
    fileInfo.dataType = RasterDataType::Float32;
    fileInfo.byteOrder = ByteOrder::LittleEndian;
    fileInfo.headerOffset = 0;

    /*
     * If the header contains DATATYPE, nrBytes
     * can still be used for compatibility.
     * Standard ESRI FLT is always Float32.
     */

    string dataFileName = fileName;

    if (extension.empty())
        dataFileName += ".flt";

    if (! readRasterFloatData(dataFileName, rasterGrid, fileInfo, errorStr))
    {
        rasterGrid->clear();
        return false;
    }

    gis::updateMinMaxRasterGrid(rasterGrid);
    rasterGrid->isLoaded = true;

    return true;
}


bool readEnviGrid(const string fileName, Crit3DRasterGrid* rasterGrid,
                  int currentUtmZone, string& errorStr)
{
    errorStr.clear();

    if (rasterGrid == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    rasterGrid->clear();

    RasterFileInfo fileInfo;

    if (! readEnviHeader(fileName, rasterGrid->header, currentUtmZone,
                        fileInfo, errorStr))
    {
        rasterGrid->clear();
        return false;
    }

    /*
     * ENVI data file is normally <name>.img.
     */
    const string dataFileName = fileName + ".img";

    if (! readRasterFloatData(dataFileName, rasterGrid, fileInfo, errorStr))
    {
        rasterGrid->clear();
        return false;
    }

    updateMinMaxRasterGrid(rasterGrid);
    rasterGrid->isLoaded = true;

    return true;
}


bool readEsriGridAscii(const string& fileName, Crit3DRasterGrid* rasterGrid, string& errorStr)
{
    errorStr.clear();

    if (rasterGrid == nullptr || rasterGrid->header == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    ifstream file(fileName);

    if (!file.is_open())
    {
        errorStr = "Wrong or missing file: " + fileName;
        return false;
    }

    bool hasCols = false;
    bool hasRows = false;
    bool hasCellSize = false;
    bool hasLLX = false;
    bool hasLLY = false;
    bool hasNoData = false;

    string line, key, value;

    /*
     * Read the header.
     *
     * The first non-header line is retained and then
     * processed as the first raster data line.
     */
    string firstDataLine;

    while (getline(file, line))
    {
        if (trim(line).empty())
            continue;

        if (!parseKeyValue(line, key, value))
        {
            firstDataLine = line;
            break;
        }

        const string upKey =
            upperCase(key);

        if (upKey == "NCOLS")
        {
            int valueInt;
            if (!parseInt(value, valueInt) || valueInt <= 0)
            {
                errorStr = "Invalid NCOLS value.";
                return false;
            }

            rasterGrid->header->nrCols = valueInt;

            hasCols = true;
        }
        else if (upKey == "NROWS")
        {
            int valueInt;

            if (!parseInt(value, valueInt) || valueInt <= 0)
            {
                errorStr = "Invalid NROWS value.";
                return false;
            }

            rasterGrid->header->nrRows = valueInt;

            hasRows = true;
        }
        else if (upKey == "XLLCORNER")
        {
            double valueDouble;
            if (!parseDouble(value, valueDouble))
            {
                errorStr = "Invalid XLLCORNER value.";
                return false;
            }

            rasterGrid->header->llCorner.x = valueDouble;

            hasLLX = true;
        }
        else if (upKey == "YLLCORNER")
        {
            double valueDouble;
            if (!parseDouble(value, valueDouble))
            {
                errorStr = "Invalid YLLCORNER value.";
                return false;
            }

            rasterGrid->header->llCorner.y = valueDouble;

            hasLLY = true;
        }
        else if (upKey == "CELLSIZE")
        {
            double valueDouble;

            if (!parseDouble(value, valueDouble) || valueDouble <= 0.)
            {
                errorStr = "Invalid CELLSIZE value.";
                return false;
            }

            rasterGrid->header->cellSize = valueDouble;

            rasterGrid->header->invCellSize = 1.0 / valueDouble;

            hasCellSize = true;
        }
        else if (upKey == "NODATA_VALUE" || upKey == "NODATA")
        {
            float valueFloat;

            if (!parseFloat(value, valueFloat))
            {
                errorStr = "Invalid NODATA value.";
                return false;
            }

            rasterGrid->header->flag = valueFloat;

            hasNoData = true;
        }
        else if (upKey == "BYTEORDER")
        {
            /*
             * ASCII data have no byte order.
             * The key is accepted for compatibility.
             */
            continue;
        }
        else if (upKey == "DATATYPE")
        {
            int nrBytes;
            if (! parseInt(value, nrBytes))
            {
                errorStr = "Invalid DATATYPE value.";
                return false;
            }

            rasterGrid->header->nrBytes = nrBytes;
        }
        else
        {
            /*
             * Unknown header key.
             *
             * Do not immediately consider it data,
             * because ESRI headers may contain additional
             * optional keys.
             */
            continue;
        }

        if (hasCols && hasRows &&
            hasCellSize &&
            hasLLX && hasLLY &&
            hasNoData)
        {
            /*
             * Standard ESRI ASCII header is complete.
             *
             * Continue reading until the first line that
             * cannot be interpreted as a key/value pair.
             */
            continue;
        }
    }

    if (!hasCols || !hasRows ||
        !hasCellSize ||
        !hasLLX || !hasLLY ||
        !hasNoData)
    {
        errorStr =
            "Missing keys in ASCII grid header.";

        return false;
    }

    if (rasterGrid->header->nrBytes == 0)
        rasterGrid->header->nrBytes = 4;

    if (!rasterGrid->initializeGrid())
    {
        errorStr =
            "Memory error: file too big.";

        return false;
    }

    int row = 0;

    auto readDataLine =
        [&](const string& dataLine) -> bool
    {
        if (row >= rasterGrid->header->nrRows)
            return true;

        istringstream stream(dataLine);

        float value;
        int col = 0;

        while (stream >> value)
        {
            if (col >= rasterGrid->header->nrCols)
            {
                errorStr =
                    "Too many values in ASCII raster row " +
                    to_string(row) + ".";

                return false;
            }

            rasterGrid->value[row][col] =
                value;

            ++col;
        }

        if (col != rasterGrid->header->nrCols)
        {
            errorStr =
                "Wrong number of values in ASCII raster row " +
                to_string(row) + ".";

            return false;
        }

        ++row;

        return true;
    };

    if (!firstDataLine.empty())
    {
        if (!readDataLine(firstDataLine))
        {
            rasterGrid->clear();
            return false;
        }
    }

    while (row < rasterGrid->header->nrRows &&
           getline(file, line))
    {
        if (trim(line).empty())
            continue;

        if (!readDataLine(line))
        {
            rasterGrid->clear();
            return false;
        }
    }

    if (row != rasterGrid->header->nrRows)
    {
        errorStr =
            "Unexpected end of ASCII raster file. " +
            to_string(
                rasterGrid->header->nrRows - row) +
            " rows are missing.";

        rasterGrid->clear();
        return false;
    }

    return true;
}


bool openRaster(const string fileName, Crit3DRasterGrid* rasterGrid, int currentUtmZone, string& errorStr)
{
    errorStr.clear();

    if (rasterGrid == nullptr)
    {
        errorStr = "Invalid raster grid.";
        return false;
    }

    if (fileName.size() <= 4)
    {
        errorStr = "Wrong filename.";
        return false;
    }

    const string extension = lowerCase(fileName.substr(fileName.size() - 4));

    const string fileNameWithoutExt = fileName.substr(0, fileName.size() - 4);

    if (extension == ".flt")
    {
        return readEsriGridFlt(fileNameWithoutExt, rasterGrid, errorStr);
    }

    if (extension == ".asc")
    {
        return readEsriGridAscii(fileName, rasterGrid, errorStr);
    }

    if (extension == ".bil")
    {
        return readEsriGridBil(fileNameWithoutExt, rasterGrid, currentUtmZone, errorStr);
    }

    if (extension == ".img")
    {
        return readEnviGrid(fileNameWithoutExt, rasterGrid, currentUtmZone, errorStr);
    }

    errorStr = "Format allowed: .flt, .asc, .bil, .img";

    return false;
}


bool writeEnviGrid(string fileName,
                   int utmZone,
                   Crit3DRasterGrid* rasterGrid,
                   string& error)
{
    error.clear();

    if (rasterGrid == nullptr ||
        rasterGrid->header == nullptr)
    {
        error = "Invalid raster grid.";
        return false;
    }

    const string imgFileName =
        fileName + ".img";

    FILE* file =
        fopen(imgFileName.c_str(), "wb");

    if (file == nullptr)
    {
        error =
            "Error writing file: " +
            imgFileName +
            "\n" +
            strerror(errno);

        return false;
    }

    const int nRows =
        rasterGrid->header->nrRows;

    const int nCols =
        rasterGrid->header->nrCols;

    for (int row = 0; row < nRows; ++row)
    {
        const size_t written =
            fwrite(
                rasterGrid->value[row],
                sizeof(float),
                static_cast<size_t>(nCols),
                file);

        if (written !=
            static_cast<size_t>(nCols))
        {
            error =
                "Error writing ENVI raster data "
                "at row " +
                to_string(row) + ".";

            fclose(file);
            return false;
        }
    }

    if (fclose(file) != 0)
    {
        error =
            "Error closing file: " +
            imgFileName;

        return false;
    }

    const string headerFileName =
        fileName + ".hdr";

    ofstream headerFile(headerFileName);

    if (!headerFile.is_open())
    {
        error =
            "Error writing file: " +
            headerFileName +
            "\n" +
            strerror(errno);

        return false;
    }

    headerFile << "ENVI\n";
    headerFile << "description = {raster grid}\n";

    headerFile
        << "samples = "
        << nCols
        << "\n";

    headerFile
        << "lines = "
        << nRows
        << "\n";

    headerFile << "bands = 1\n";
    headerFile << "header offset = 0\n";
    headerFile << "file type = ENVI Standard\n";
    headerFile << "data type = 4\n";
    headerFile << "interleave = bsq\n";
    headerFile << "byte order = 0\n";

    headerFile
        << "data ignore value = "
        << rasterGrid->header->flag
        << "\n";

    const double yTopLeftCorner =
        rasterGrid->header->llCorner.y +
        static_cast<double>(nRows) *
            rasterGrid->header->cellSize;

    headerFile
        << "map info = {UTM, 1, 1, "
        << rasterGrid->header->llCorner.x
        << ", "
        << yTopLeftCorner
        << ", "
        << rasterGrid->header->cellSize
        << ", "
        << rasterGrid->header->cellSize
        << ", "
        << utmZone
        << ", North, WGS-84, units=Meters}\n";

    if (!headerFile.good())
    {
        error =
            "Error writing file: " +
            headerFileName;

        return false;
    }

    return true;
}


bool getGeoExtentsFromUTMHeader(
    const Crit3DGisSettings& mySettings,
    Crit3DRasterHeader* utmHeader,
    Crit3DLatLonHeader* latLonHeader)
{
    if (utmHeader == nullptr ||
        latLonHeader == nullptr)
    {
        return false;
    }

    Crit3DGeoPoint vertices[4];

    Crit3DUtmPoint vertex =
        utmHeader->llCorner;

    /*
     * Lower-left
     */
    getLatLonFromUtm(
        mySettings,
        vertex,
        vertices[0]);

    /*
     * Lower-right
     */
    vertex.x +=
        static_cast<double>(
            utmHeader->nrCols) *
        utmHeader->cellSize;

    getLatLonFromUtm(
        mySettings,
        vertex,
        vertices[1]);

    /*
     * Upper-right
     */
    vertex.y +=
        static_cast<double>(
            utmHeader->nrRows) *
        utmHeader->cellSize;

    getLatLonFromUtm(
        mySettings,
        vertex,
        vertices[2]);

    /*
     * Upper-left
     */
    vertex.x =
        utmHeader->llCorner.x;

    getLatLonFromUtm(
        mySettings,
        vertex,
        vertices[3]);

    Crit3DGeoPoint LLcorner;
    Crit3DGeoPoint URcorner;

    LLcorner.longitude =
        min(vertices[0].longitude,
            vertices[3].longitude);

    URcorner.longitude =
        max(vertices[1].longitude,
            vertices[2].longitude);

    if (mySettings.startLocation.latitude >= 0)
    {
        LLcorner.latitude =
            min(vertices[0].latitude,
                vertices[1].latitude);

        URcorner.latitude =
            max(vertices[2].latitude,
                vertices[3].latitude);
    }
    else
    {
        LLcorner.latitude =
            max(vertices[0].latitude,
                vertices[1].latitude);

        URcorner.latitude =
            min(vertices[2].latitude,
                vertices[3].latitude);
    }

    latLonHeader->nrRows =
        utmHeader->nrRows;

    latLonHeader->nrCols =
        utmHeader->nrCols;

    latLonHeader->dx =
        (URcorner.longitude -
         LLcorner.longitude) /
        latLonHeader->nrCols;

    latLonHeader->dy =
        (URcorner.latitude -
         LLcorner.latitude) /
        latLonHeader->nrRows;

    latLonHeader->llCorner.latitude =
        LLcorner.latitude;

    latLonHeader->llCorner.longitude =
        LLcorner.longitude;

    latLonHeader->flag =
        utmHeader->flag;

    return true;
}


bool getGeoExtentsFromLatLonHeader(
    const Crit3DGisSettings& mySettings,
    double cellSize,
    Crit3DRasterHeader* utmHeader,
    Crit3DLatLonHeader* latLonHeader)
{
    if (utmHeader == nullptr ||
        latLonHeader == nullptr ||
        cellSize <= 0.)
    {
        return false;
    }

    Crit3DUtmPoint vertices[4];

    Crit3DGeoPoint geoPoint;

    /*
     * Lower-left
     */
    geoPoint.latitude =
        latLonHeader->llCorner.latitude;

    geoPoint.longitude =
        latLonHeader->llCorner.longitude;

    getUtmFromLatLon(
        mySettings.utmZone,
        geoPoint,
        &vertices[0]);

    /*
     * Lower-right
     */
    geoPoint.longitude =
        latLonHeader->llCorner.longitude +
        static_cast<double>(
            latLonHeader->nrCols) *
            latLonHeader->dx;

    getUtmFromLatLon(
        mySettings.utmZone,
        geoPoint,
        &vertices[1]);

    /*
     * Upper-right
     */
    geoPoint.latitude =
        latLonHeader->llCorner.latitude +
        static_cast<double>(
            latLonHeader->nrRows) *
            latLonHeader->dy;

    getUtmFromLatLon(
        mySettings.utmZone,
        geoPoint,
        &vertices[2]);

    /*
     * Upper-left
     */
    geoPoint.longitude =
        latLonHeader->llCorner.longitude;

    getUtmFromLatLon(
        mySettings.utmZone,
        geoPoint,
        &vertices[3]);

    const double xmin =
        floor(min(vertices[0].x,
                  vertices[3].x));

    const double xmax =
        floor(max(vertices[1].x,
                  vertices[2].x)) + 1.;

    const double ymin =
        floor(min(vertices[0].y,
                  vertices[1].y));

    const double ymax =
        floor(max(vertices[2].y,
                  vertices[3].y)) + 1.;

    utmHeader->cellSize =
        cellSize;

    utmHeader->invCellSize =
        1.0 / cellSize;

    utmHeader->nrCols =
        static_cast<int>(
            floor(
                (xmax - xmin) /
                cellSize) + 1);

    utmHeader->nrRows =
        static_cast<int>(
            floor(
                (ymax - ymin) /
                cellSize) + 1);

    utmHeader->llCorner.x =
        xmin;

    utmHeader->llCorner.y =
        ymin;

    utmHeader->flag =
        latLonHeader->flag;

    return true;
}


double getGeoCellSizeFromLatLonHeader(
    const Crit3DGisSettings& mySettings,
    Crit3DLatLonHeader* latLonHeader)
{
    if (latLonHeader == nullptr ||
        latLonHeader->nrRows <= 0 ||
        latLonHeader->nrCols <= 0)
    {
        return NODATA;
    }

    Crit3DUtmPoint vertices[4];

    Crit3DGeoPoint geoPoint;

    /*
     * Lower-left
     */
    geoPoint.latitude =
        latLonHeader->llCorner.latitude;

    geoPoint.longitude =
        latLonHeader->llCorner.longitude;

    getUtmFromLatLon(
        mySettings.utmZone,
        geoPoint,
        &vertices[0]);

    /*
     * Lower-right
     */
    geoPoint.longitude =
        latLonHeader->llCorner.longitude +
        static_cast<double>(
            latLonHeader->nrCols) *
            latLonHeader->dx;

    getUtmFromLatLon(
        mySettings.utmZone,
        geoPoint,
        &vertices[1]);

    /*
     * Upper-right
     */
    geoPoint.latitude =
        latLonHeader->llCorner.latitude +
        static_cast<double>(
            latLonHeader->nrRows) *
            latLonHeader->dy;

    getUtmFromLatLon(
        mySettings.utmZone,
        geoPoint,
        &vertices[2]);

    /*
     * Upper-left
     */
    geoPoint.longitude =
        latLonHeader->llCorner.longitude;

    getUtmFromLatLon(
        mySettings.utmZone,
        geoPoint,
        &vertices[3]);

    const double xCellSize =
        (vertices[1].x -
         vertices[0].x) /
        latLonHeader->nrCols;

    const double yCellSize =
        (vertices[3].y -
         vertices[0].y) /
        latLonHeader->nrRows;

    return min(
        fabs(xCellSize),
        fabs(yCellSize));
}


} // namespace gis