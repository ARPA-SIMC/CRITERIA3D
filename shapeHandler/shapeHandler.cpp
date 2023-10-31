/*!
    \file shapeHandler.cpp

    \abstract shapefile handler

    This file is part of CRITERIA-3D distribution.

    CRITERIA-3D has been developed by A.R.P.A.E. Emilia-Romagna.

    \copyright
    CRITERIA-3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    CRITERIA-3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA-3D.  If not, see <http://www.gnu.org/licenses/>.

    \authors
    Laura Costantini laura.costantini0@gmail.com
    Fausto Tomei ftomei@arpae.it
*/

#include "basicMath.h"
#include "shapeHandler.h"
#include "commonConstants.h"
#include <fstream>
#include <string.h>


Crit3DShapeHandler::Crit3DShapeHandler()
    : m_handle(nullptr), m_dbf(nullptr), m_count(0), m_type(0)
{ }


Crit3DShapeHandler::~Crit3DShapeHandler()
{
    close();
}


void Crit3DShapeHandler::close()
{
    if ((m_handle != nullptr) || (m_dbf != nullptr))
    {
        SHPClose(m_handle);
        DBFClose(m_dbf);
    }

    m_fieldsList.clear();
    m_fieldsTypeList.clear();
    holes.clear();

    m_handle = nullptr;
    m_dbf = nullptr;
}


void Crit3DShapeHandler::closeDBF()
{
    if (m_dbf != nullptr)
    {
        DBFClose(m_dbf);
    }
    m_dbf = nullptr;
}

void Crit3DShapeHandler::closeSHP()
{
    if ((m_handle != nullptr))
    {
        SHPClose(m_handle);
    }
    m_handle = nullptr;
}


bool Crit3DShapeHandler::open(std::string filename)
{
    close();
    m_handle = SHPOpen(filename.c_str(), "r+b");
    m_dbf = DBFOpen(filename.c_str(), "r+b");
    m_filepath = filename;
    if ( (m_handle == nullptr) || (m_dbf == nullptr)) return false;

    SHPGetInfo(m_handle, &m_count, &m_type, nullptr, nullptr);
    m_fields = m_dbf->nFields;

    char *fieldNamePtr =  new char[XBASE_FLDNAME_LEN_READ];
    DBFFieldType fieldType;

    m_fieldsList.clear();
    m_fieldsTypeList.clear();
    for (int i = 0; i < m_fields; i++)
    {
        fieldType = DBFGetFieldInfo( m_dbf, i, fieldNamePtr, nullptr, nullptr);
        m_fieldsList.push_back(std::string(fieldNamePtr));
        m_fieldsTypeList.push_back(fieldType);
    }

    // check if WGS84 PROJ and set UTM
    std::string filePrj = filename;
    std::string::size_type i = filePrj.rfind('.', filePrj.length());
    std::string prjExt = "prj";
    if (i != std::string::npos)
    {
        filePrj.replace(i+1, prjExt.length(), prjExt);
    }
    isWGS84Proj(filePrj);
    setUTMzone(filePrj);

    // save holes inside parts
    ShapeObject myShape;
    Point<double> point;

    m_parts = 0;
    m_holes = 0;
    holes.clear();
    holes.resize(unsigned(m_count));

    std::vector<ShapeObject::Part> shapeParts;

    for (unsigned int i = 0; i < unsigned(m_count); i++)
    {
        getShape(int(i), myShape);
        shapeParts = myShape.getParts();

        unsigned int nrParts = myShape.getPartCount();
        m_parts += nrParts;

        holes[i].resize(nrParts);

        for (unsigned int j = 0; j < nrParts; j++)
        {
            // holes
            if (shapeParts[j].hole)
            {
                m_holes++;
                // check first point
                unsigned long offset = shapeParts[j].offset;
                point = myShape.getVertex(offset);
                int index = myShape.getIndexPart(point.x, point.y);
                if (index != NODATA)
                {
                    holes[i][unsigned(index)].push_back(j);
                }
            }
        }
        shapeParts.clear();
    }

    return true;
}


void Crit3DShapeHandler::newShapeFile(std::string filename, int nShapeType)
{
    m_handle = SHPCreate(filename.c_str(), nShapeType);
    m_dbf = DBFCreate(filename.c_str());
    m_filepath = filename;
}


bool Crit3DShapeHandler::openDBF(std::string filename)
{
    closeDBF();
    m_dbf = DBFOpen(filename.c_str(), "r+b");
    if (m_dbf == nullptr) return false;

    m_fields = m_dbf->nFields;

    char *fieldName =  new char[XBASE_FLDNAME_LEN_READ+1];
    DBFFieldType fieldType;

    m_fieldsList.clear();
    m_fieldsTypeList.clear();

    for (int i = 0; i<m_fields; i++)
    {
        fieldType = DBFGetFieldInfo( m_dbf, i, fieldName, nullptr, nullptr);
        m_fieldsList.push_back(std::string(fieldName));
        m_fieldsTypeList.push_back(fieldType);
    }

    return true;
}


bool Crit3DShapeHandler::openSHP(std::string filename)
{
    closeSHP();
    m_handle = SHPOpen(filename.c_str(), "r+b");
    if (m_handle == nullptr) return false;

    SHPGetInfo(m_handle, &m_count, &m_type, nullptr, nullptr);
    return true;
}


bool Crit3DShapeHandler::isWGS84Proj(std::string prjFileName)
{
    std::ifstream prjFile (prjFileName);
    std::string line;
    std::string proj = "WGS_1984";

    if (prjFile.is_open())
    {
        while ( getline (prjFile,line) )
        {
            if (line.find(proj) != std::string::npos)
            {
                m_isWGS84 = true;
                prjFile.close();
                return true;
            }
            else
            {
                m_isWGS84 = false;
                prjFile.close();
                return false;
            }
        }
    }
    else
    {
        m_isWGS84 = false;
        return false;
    }

    return false;
}


bool Crit3DShapeHandler::setUTMzone(std::string prjFileName)
{
    std::ifstream prjFile (prjFileName);
    std::string line;
    std::string utmString = "UTM_Zone_";
    std::string separator = ",";

    if (prjFile.is_open())
    {
        while ( getline (prjFile,line) )
        {
            std::size_t found = line.find(utmString);
            if (found != std::string::npos)
            {
                std::size_t start = found + 9;
                std::size_t foundEnd = line.find(separator);
                std::string utm = line.substr(start, foundEnd-1-start);
                m_utmZone = std::stoi(utm);
                std::string emisphere = utm.substr(utm.length()-1, utm.length()-1);
                m_isNorth = (emisphere == "N");
                prjFile.close();
                return true;
            }
            else
            {
                prjFile.close();
                return false;
            }
        }
    }

    return false;
}


bool Crit3DShapeHandler::getShape(int index, ShapeObject &shape)
{
    if ( (m_handle == nullptr) || (m_dbf == nullptr)) return false;

    SHPObject *obj = SHPReadObject(m_handle, index);
    shape.assign(obj);
    SHPDestroyObject(obj);

    return true;
}


int	Crit3DShapeHandler::getDBFFieldIndex(const char *pszFieldName)
{
    return DBFGetFieldIndex(m_dbf, pszFieldName);
}

int	Crit3DShapeHandler::isDBFRecordDeleted(int record)
{
    return DBFIsRecordDeleted(m_dbf, record);
}


int	Crit3DShapeHandler::getFieldPos(std::string fieldName)
{
    for (int i = 0; i < m_fields; i++)
        if (m_fieldsList.at(unsigned(i)) == fieldName)
            return i;

    return -1;
}


bool Crit3DShapeHandler::existField(std::string fieldName)
{
    for (int i = 0; i < m_fields; i++)
        if (m_fieldsList.at(unsigned(i)) == fieldName)
            return true;

    return false;
}


DBFFieldType Crit3DShapeHandler::getFieldType(std::string fieldName)
{
    int pos = getFieldPos(fieldName);

    if (pos == -1)
        return FTInvalid;
    else
        return getFieldType(pos);
}


double Crit3DShapeHandler::getNumericValue(int shapeNumber, std::string fieldName)
{
    int fieldPos = getFieldPos(fieldName);

    if (fieldPos == -1) return NODATA;

    return getNumericValue(shapeNumber, fieldPos);
}


double Crit3DShapeHandler::getNumericValue(int shapeNumber, int fieldPos)
{
    DBFFieldType fieldType = getFieldType(fieldPos);

    double value = NODATA;
    if (fieldType == FTInteger)
    {
        value = readIntAttribute(shapeNumber, fieldPos);
    }
    else if (fieldType == FTDouble)
    {
        value = readDoubleAttribute(shapeNumber, fieldPos);
    }

    // check zero as nodata
    if (isEqual(value, 0))
    {
        std::string strValue = readStringAttribute(shapeNumber, fieldPos);
        if (strValue == "" || strValue == "******")
        {
            value = NODATA;
        }
    }

    return value;
}


std::string Crit3DShapeHandler::getStringValue(int shapeNumber, std::string fieldName)
{
    int fieldPos = getFieldPos(fieldName);
    if (fieldPos == -1) return "";

    return readStringAttribute(shapeNumber, fieldPos);
}

int Crit3DShapeHandler::readIntAttribute(int shapeNumber, int fieldPos)
{
    return DBFReadIntegerAttribute(m_dbf,shapeNumber,fieldPos);
}

bool Crit3DShapeHandler::writeIntAttribute(int shapeNumber, int fieldPos, int nFieldValue)
{
    return DBFWriteIntegerAttribute(m_dbf,shapeNumber,fieldPos,nFieldValue);
}

double Crit3DShapeHandler::readDoubleAttribute(int shapeNumber, int fieldPos)
{
    return DBFReadDoubleAttribute(m_dbf,shapeNumber,fieldPos);
}

bool Crit3DShapeHandler::writeDoubleAttribute(int shapeNumber, int fieldPos, double dFieldValue)
{
    return DBFWriteDoubleAttribute(m_dbf,shapeNumber,fieldPos,dFieldValue);
}

std::string Crit3DShapeHandler::readStringAttribute(int shapeNumber, int fieldPos)
{
    return DBFReadStringAttribute(m_dbf,shapeNumber,fieldPos);
}

bool Crit3DShapeHandler::writeStringAttribute(int shapeNumber, int fieldPos, const char* pszFieldValue)
{
    return DBFWriteStringAttribute(m_dbf,shapeNumber,fieldPos, pszFieldValue);
}

bool Crit3DShapeHandler::deleteRecord(int shapeNumber)
{
    return DBFMarkRecordDeleted(m_dbf,shapeNumber,true);
}

/*
bool Crit3DShapeHandler::addRecord(std::vector<std::string> fields)
{
    if( DBFGetFieldCount(m_dbf) != fields.size() )
    {
        return false;
    }

    int iRecord = DBFGetRecordCount( m_dbf );
    int typeField;

    // --------------------------------------------------------------------
    //	Loop assigning the new field values.
    // --------------------------------------------------------------------

    for( int i = 0; i < DBFGetFieldCount(m_dbf); i++ )
    {
        typeField = getFieldType(i);
        if( fields.at(i) == "")
        {
            DBFWriteNULLAttribute(m_dbf, iRecord, i );
        }
        else if( typeField == FTString )
        {
            DBFWriteStringAttribute(m_dbf, iRecord, i, fields.at(i).c_str() );
        }
        else if (typeField == FTInteger)
        {
            DBFWriteIntegerAttribute(m_dbf, iRecord, i, std::stoi(fields.at(i))) ;
        }
        else if (typeField == FTDouble)
        {
            DBFWriteDoubleAttribute(m_dbf, iRecord, i, std::stod(fields.at(i)) );
        }

    }

    return true;
}
*/

// LC MAI testata
bool Crit3DShapeHandler::addShape(std::string type, std::vector<double> coordinates)
{
    openSHP(m_filepath);
    if ( (m_handle == nullptr) || (m_dbf == nullptr)) return false;
    // shpadd shp_file [[x y] [+]]

    int shpType = 5; // SHPT_POLYGON

    if (type != "Polygon")
    {
        return false;
    }
    // --------------------------------------------------------------------
    //	Build a vertex/part list
    // --------------------------------------------------------------------
        int		nVertices, *panParts;
        double	*padfX, *padfY;
        SHPObject	*psObject;

        unsigned int sizeCoord = unsigned(coordinates.size());

        padfX = (double *) malloc(sizeof(double) * sizeCoord);
        padfY = (double *) malloc(sizeof(double) * sizeCoord);

        nVertices = 0;

        panParts = (int *) malloc(sizeof(int) * sizeCoord );

        int nParts = 1;
        panParts[0] = 0;

        for( unsigned int i = 0; i < sizeCoord;  i++)
        {
            if (i%2 == 0) // X coord
            {
                padfX[nVertices] = coordinates[i];
            }
            else // Y coord
            {
                padfY[nVertices] = coordinates[i];
                nVertices += 1;
                panParts[nParts] = nVertices;
                nParts++;
            }
        }

    // --------------------------------------------------------------------
    //      Write the new entity to the shape file.
    // --------------------------------------------------------------------
        //psObject = SHPCreateObject( shpType, iShape, nParts, panParts, NULL, nVertices, padfX, padfY, NULL, NULL );

        psObject = SHPCreateSimpleObject( shpType, nVertices, padfX, padfY, NULL);
        int ret = SHPWriteObject( m_handle, -1, psObject );
        SHPDestroyObject( psObject );

        closeSHP();
        free( panParts );
        free( padfX );
        free( padfY );

        return (ret != -1);
}


bool Crit3DShapeHandler::addField(const char * fieldName, int type, int nWidth, int nDecimals )
{

    DBFFieldType DBFtype;
    if (type == 0)
    {
        DBFtype = FTString;
    }
    else if (type == 1)
    {
        DBFtype = FTInteger;
    }
    else if (type == 2)
    {
        DBFtype = FTDouble;
    }
    else
    {
        DBFtype = FTInvalid;
    }

    if (DBFAddField(m_dbf, fieldName, DBFtype, nWidth, nDecimals) != -1)
    {
        m_fields = m_fields + 1;
        m_fieldsList.push_back(std::string(fieldName));
        m_fieldsTypeList.push_back(DBFtype);
        return true;
    }
    else
    {
        return false;
    }
}


bool Crit3DShapeHandler::removeField(int iField)
{
    if (DBFDeleteField(m_dbf, iField))
    {
        m_fields = m_fields - 1;
        m_fieldsList.erase(m_fieldsList.begin()+iField);
        m_fieldsTypeList.erase(m_fieldsTypeList.begin()+iField);
        return true;
    }
    else
    {
        return false;
    }
}


void Crit3DShapeHandler::packDBF(std::string newFile)
{
    DBFHandle hDBF;
    hDBF = DBFCreate(newFile.c_str());

    // copy fields
    for(unsigned int i = 0; i < unsigned(m_fields); i++ )
    {
        int nWidth = m_dbf->panFieldSize[i];
        int nDecimals = m_dbf->panFieldDecimals[i];
        DBFAddField( hDBF, m_fieldsList.at(i).c_str(), m_fieldsTypeList.at(i), nWidth, nDecimals );
    }

    //copy records if not deleted
    int nRecord = DBFGetRecordCount(m_dbf);
    int newCount = 0;
    for (int i = 0; i<nRecord; i++)
    {
        if (!DBFIsRecordDeleted(m_dbf, i))
        {
            const char * row = DBFReadTuple(m_dbf, i );

            DBFWriteTuple(hDBF, newCount, (void*)row);
            newCount = newCount + 1;
        }

    }
    DBFClose( hDBF );

}

bool Crit3DShapeHandler::existRecordDeleted()
{
    int nRecord = DBFGetRecordCount(m_dbf);

    for (int i = 0; i<nRecord; i++)
    {
        if (DBFIsRecordDeleted(m_dbf, i))
        {
            return true;
        }
    }

    return false;
}


void Crit3DShapeHandler::packSHP(std::string newFile)
{
    SHPHandle hSHP;
    hSHP = SHPCreate(newFile.c_str(), m_type);

    for( int i = 0; i < m_count; i++ )
    {
        if (!DBFIsRecordDeleted(m_dbf, i))
        {
            SHPObject *obj = SHPReadObject(m_handle, i);
            SHPWriteObject(hSHP, -1, obj);
            SHPDestroyObject(obj);
        }
    }

    SHPClose(hSHP);
}


std::vector<unsigned int> Crit3DShapeHandler::getHoles(int shapeNumber, int partNumber)
{
    if (shapeNumber > m_count || partNumber > int(holes[shapeNumber].size()))
    {
        std::vector<unsigned int> emptyVector;
        return emptyVector;
    }
    return holes[unsigned(shapeNumber)][unsigned(partNumber)];
}



int Crit3DShapeHandler::getShapeIndexfromPoint(double utmX, double utmY)
{
    if (m_handle == nullptr || m_count <= 0)
        return NODATA;

    ShapeObject myShape;
    for (int index = 0; index < m_count; index++)
    {
        if (getShape(index, myShape))
        {
            if (myShape.pointInPolygon(utmX, utmY))
            {
                return index;
            }
        }
    }

    return NODATA;
}


/* return list of attributes */
std::string Crit3DShapeHandler::getAttributesList(int index)
{
    // check index
    if (index == NODATA || index >= m_count)
    {
        return "";
    }

    std::string shapeData = "Shape nr. " + std::to_string(index) + "\n";

    if (m_fields > 0)
    {
        shapeData += "\nAttributes:\n";

        for (int i = 0; i < m_fields; i++)
        {
            std::string nameField = getFieldName(i);
            int typeField = getFieldType(i);

            std::string myStr;
            if (typeField == FTString)
            {
                myStr = readStringAttribute(index,i);
            }
            else if (typeField == FTInteger)
            {
                myStr = std::to_string(readIntAttribute(index,i));
            }
            else if (typeField == FTDouble)
            {
                myStr = std::to_string(readDoubleAttribute(index,i));
            }
            else
            {
                myStr = "invalid field type ";
            }

            shapeData += nameField + " = " + myStr + "\n";
        }
    }

    return shapeData;
}

