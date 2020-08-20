#include <float.h>
#include <algorithm>
#include <math.h>

#include "commonConstants.h"
#include "zonalStatistic.h"
#include "shapeToRaster.h"

std::vector <std::vector<int> > computeMatrixAnalysis(Crit3DShapeHandler* shapeRef, Crit3DShapeHandler* shapeVal,
                          gis::Crit3DRasterGrid *rasterRef, gis::Crit3DRasterGrid *rasterVal,
                          std::string* error)
{
    /*
     * move check before call
    // check shape type
    if ( shapeRef->getTypeString() != shapeVal->getTypeString() || shapeRef->getTypeString() != "2D Polygon" )
    {
        *error = "shape type error: not 2D Polygon type" ;
        return false;
    }

    // check proj
    if (shapeRef->getIsWGS84() == false || shapeVal->getIsWGS84() == false)
    {
        *error = "projection error: not WGS84" ;
        return false;
    }

    // check utm zone
    if (shapeRef->getUtmZone() != shapeVal->getUtmZone())
    {
        *error = "utm zone: different utm zones" ;
        return false;
    }
    */
    unsigned int nrRefShapes = unsigned(shapeRef->getShapeCount());
    unsigned int nrValShapes = unsigned(shapeVal->getShapeCount());

    // analysis matrix
   std::vector <std::vector<int> > matrix(nrRefShapes, std::vector<int>(nrValShapes));
   std::vector <int> vectorNull(nrRefShapes, 0);

   for (int row = 0; row < rasterRef->header->nrRows; row++)
   {
       for (int col = 0; col < rasterRef->header->nrCols; col++)
       {
           int refIndex = int(rasterRef->value[row][col]);
           int valIndex = int(rasterVal->value[row][col]);

           if (refIndex != NODATA)
           {
               if (valIndex != NODATA)
                   matrix[unsigned(refIndex)][unsigned(valIndex)]++;
               else
                   vectorNull[unsigned(refIndex)]++;
           }
       }
   }
   return matrix;
}

bool zonalStatisticsShape(Crit3DShapeHandler* shapeRef, Crit3DShapeHandler* shapeVal,
                          std::vector <std::vector<int> > matrix, std::vector <int> vectorNull,
                          std::string valField, std::string valFieldOutput, opType aggregationType,
                          std::string* error)
{

    //check if valField exists
    int fieldIndex = shapeVal->getDBFFieldIndex(valField.c_str());
    if (fieldIndex == -1)
    {
        *error = shapeVal->getFilepath() + "has not field called " + valField.c_str();
        return false;
    }

    // add new field to shapeRef
    DBFFieldType fieldType = shapeVal->getFieldType(fieldIndex);
    shapeRef->addField(valFieldOutput.c_str(), fieldType, shapeVal->nWidthField(fieldIndex), shapeVal->nDecimalsField(fieldIndex));

    unsigned int nrRefShapes = unsigned(shapeRef->getShapeCount());
    unsigned int nrValShapes = unsigned(shapeVal->getShapeCount());
    double value = 0.0;
    double validPoint = 0.0;
    double sumValues = 0;
    std::vector<double> validPoints;
    std::vector <int> aggregationResults(nrRefShapes, 0);
    int nValues = 0;

    for (unsigned int row = 0; row < nrRefShapes; row++)
    {
        for (unsigned int col = 0; col < nrValShapes; col++)
        {
            if (matrix[row][col] > 0)
            {
                if (fieldType == FTInteger)
                {
                    value = (double)shapeVal->readIntAttribute(col,fieldIndex);
                }
                else if (fieldType == FTDouble)
                {
                    value = shapeVal->readDoubleAttribute(col,fieldIndex);
                }
                if (value == NODATA)
                {
                    vectorNull[row] = vectorNull[row] + matrix[row][col];
                }
                else if (aggregationType == AVG)
                {
                    validPoint = validPoint + matrix[row][col];
                    sumValues = sumValues + value*matrix[row][col];
                    nValues = nValues + 1;
                }
            }
        }
        aggregationResults.push_back(sumValues/nValues);
        validPoints.push_back(validPoint);
        //reset
        validPoint = 0;
        sumValues = 0;
        nValues = 0;
    }

    // save value of the new field
    // each row of matrix is a shape of shapeRef
    double valueToSave = 0.0;
    for (unsigned int shapeIndex = 0; shapeIndex < nrRefShapes; shapeIndex++)
    {
        if (validPoints[shapeIndex] < vectorNull[shapeIndex])
        {
            valueToSave = NODATA;
        }
        else
        {
            valueToSave = aggregationResults[shapeIndex];
        }
        if (fieldType == FTInteger)
        {
            shapeRef->writeIntAttribute(shapeIndex, shapeRef->getDBFFieldIndex(valFieldOutput.c_str()), (int)valueToSave);
        }
        else if (fieldType == FTDouble)
        {
            shapeRef->writeDoubleAttribute(shapeIndex, shapeRef->getDBFFieldIndex(valFieldOutput.c_str()), valueToSave);
        }
    }

    // close and re-open to write also the last shape
    shapeRef->close();
    shapeRef->open(shapeRef->getFilepath());

    return true;
}

bool zonalStatisticsShapeMajority(Crit3DShapeHandler* shapeRef, Crit3DShapeHandler* shapeVal,
                          std::vector <std::vector<int> > matrix, std::vector <int> vectorNull,
                          std::string valField, std::string valFieldOutput, opType aggregationType,
                          std::string* error)
{
    // TO DO
}

/////////////////// OLD VERSION ////////////////////////////////////
bool zonalStatisticsShape(Crit3DShapeHandler* shapeRef, Crit3DShapeHandler* shapeVal,
                          gis::Crit3DRasterGrid *rasterRef, gis::Crit3DRasterGrid *rasterVal,
                          std::string valField, std::string valFieldOutput, opType aggregationType,
                          std::string* error)
{
    // check shape type
    if ( shapeRef->getTypeString() != shapeVal->getTypeString() || shapeRef->getTypeString() != "2D Polygon" )
    {
        *error = "shape type error: not 2D Polygon type" ;
        return false;
    }

    // check proj
    if (shapeRef->getIsWGS84() == false || shapeVal->getIsWGS84() == false)
    {
        *error = "projection error: not WGS84" ;
        return false;
    }

    // check utm zone
    if (shapeRef->getUtmZone() != shapeVal->getUtmZone())
    {
        *error = "utm zone: different utm zones" ;
        return false;
    }

    //check if valField exists
    int fieldIndex = shapeVal->getDBFFieldIndex(valField.c_str());
    if (fieldIndex == -1)
    {
        *error = shapeVal->getFilepath() + "has not field called " + valField.c_str();
        return false;
    }

    unsigned int nrRefShapes = unsigned(shapeRef->getShapeCount());

    // add new field to shapeRef
    DBFFieldType fieldType = shapeVal->getFieldType(fieldIndex);
    shapeRef->addField(valFieldOutput.c_str(), fieldType, shapeVal->nWidthField(fieldIndex), shapeVal->nDecimalsField(fieldIndex));

    // LC da fare restyle codice affinchè gestisca i 3 casi
    std::vector<std::string> varFieldVectorString;
    std::vector<double> varFieldVectorDouble;
    std::vector<int> varFieldVectorInt;
    size_t varFieldVectorSize = 0;

    if (fieldType == FTString)
    {
        // fill varFieldVector
        for (int record = 0; record < shapeVal->getDBFRecordCount(); record++)
        {
            std::string value = shapeVal->readStringAttribute(record,fieldIndex);
            if (std::find (varFieldVectorString.begin(), varFieldVectorString.end(), value) == varFieldVectorString.end())
            {
                varFieldVectorString.push_back(value);
            }
        }
        varFieldVectorSize = varFieldVectorString.size();

        // assign varFieldVector index to each pixel of rasterVal
        for (int row = 0; row < rasterVal->header->nrRows; row++)
        {
            for (int col = 0; col < rasterVal->header->nrCols; col++)
            {
                int shape = int(rasterVal->value[row][col]);
                if (shape!= NODATA)
                {
                    std::string valueField = shapeVal->readStringAttribute(shape,fieldIndex);
                    int vectorFieldPos = int(std::distance(varFieldVectorString.begin(), std::find (varFieldVectorString.begin(), varFieldVectorString.end(), valueField)));
                    //replace value
                    rasterVal->value[row][col] = float(vectorFieldPos);
                }
            }
        }
    }
    else if (fieldType == FTInteger)
    {

        for (int record = 0; record < shapeVal->getDBFRecordCount(); record++)
        {
            int value = shapeVal->readIntAttribute(record,fieldIndex);
            if (std::find (varFieldVectorInt.begin(), varFieldVectorInt.end(), value) == varFieldVectorInt.end())
            {
                varFieldVectorInt.push_back(value);
            }
        }

        varFieldVectorSize = varFieldVectorInt.size();

        for (int row = 0; row < rasterVal->header->nrRows; row++)
        {
            for (int col = 0; col < rasterVal->header->nrCols; col++)
            {
                int shape = int(rasterVal->value[row][col]);
                if (shape!= NODATA)
                {
                    int valueField = shapeVal->readIntAttribute(shape,fieldIndex);
                    int vectorFieldPos = int(std::distance(varFieldVectorInt.begin(), std::find (varFieldVectorInt.begin(), varFieldVectorInt.end(), valueField)));
                    //replace value
                    rasterVal->value[row][col] = float(vectorFieldPos);
                }
            }
        }
    }
    else if (fieldType == FTDouble)
    {

        for (int record = 0; record < shapeVal->getDBFRecordCount(); record++)
        {
            double value = shapeVal->readDoubleAttribute(record,fieldIndex);
            if (std::find (varFieldVectorDouble.begin(), varFieldVectorDouble.end(), value) == varFieldVectorDouble.end())
            {
                varFieldVectorDouble.push_back(value);
            }
        }

        varFieldVectorSize = varFieldVectorDouble.size();

        for (int row = 0; row < rasterVal->header->nrRows; row++)
        {
            for (int col = 0; col < rasterVal->header->nrCols; col++)
            {
                int shape = int(rasterVal->value[row][col]);
                if (shape!= NODATA)
                {
                    double valueField = shapeVal->readDoubleAttribute(shape,fieldIndex);
                    int vectorFieldPos = std::distance(varFieldVectorDouble.begin(), std::find (varFieldVectorDouble.begin(), varFieldVectorDouble.end(), valueField));
                    //replace value
                    rasterVal->value[row][col] = vectorFieldPos;
                }
            }
        }
    }

     // analysis matrix
    std::vector <std::vector<int> > matrix(nrRefShapes, std::vector<int>(varFieldVectorSize, 0));
    std::vector <int> matrixNull(nrRefShapes, 0);

    for (int row = 0; row < rasterRef->header->nrRows; row++)
    {
        for (int col = 0; col < rasterRef->header->nrCols; col++)
        {
            int refIndex = int(rasterRef->value[row][col]);
            int valIndex = int(rasterVal->value[row][col]);

            if (refIndex != NODATA)
            {
                if (valIndex != NODATA)
                    matrix[unsigned(refIndex)][unsigned(valIndex)]++;
                else
                    matrixNull[unsigned(refIndex)]++;
            }
        }
    }

    // save right matrix element for each shape
    std::vector<int> indexVector;
    int maxValue = 0;

    if (aggregationType == MAJORITY)
    {
        for (unsigned int row = 0; row < nrRefShapes; row++)
        {
            indexVector.push_back(NODATA);
            maxValue = 0;
            for (unsigned int col = 0; col < varFieldVectorSize; col++)
            {
                if (matrix[row][col] > maxValue)
                {
                    indexVector[row] = signed(col);
                    maxValue = matrix[row][col];
                }
            }
            // check on null values
            if (maxValue < matrixNull[row])
            {
                indexVector[row] = NODATA;
            }
        }
    }
    else if (aggregationType == MIN)
    {
        for (unsigned int row = 0; row < nrRefShapes; row++)
        {
            // TO DO
        }
    }
    else if (aggregationType == MAX)
    {
        for (unsigned int row = 0; row < nrRefShapes; row++)
        {
            // TO DO
        }
    }
    else if (aggregationType == AVG)
    {
        for (unsigned int row = 0; row < nrRefShapes; row++)
        {
            // TO DO
        }
    }

    // save value of the new field
    // each row of matrix is a shape of shapeRef
    for (unsigned int shapeIndex = 0; shapeIndex < nrRefShapes; shapeIndex++)
    {
        if (fieldType == FTString)
        {
            std::string varFieldFound;
            if (indexVector[shapeIndex] == NODATA)
                varFieldFound = "";
            else
                varFieldFound = varFieldVectorString[unsigned(indexVector[shapeIndex])];

            shapeRef->writeStringAttribute(shapeIndex, shapeRef->getDBFFieldIndex(valFieldOutput.c_str()), varFieldFound.c_str());
        }
        else if (fieldType == FTInteger)
        {
            int varFieldFound;
            if (indexVector[shapeIndex] == NODATA)
                varFieldFound = NODATA;
            else
                varFieldFound = varFieldVectorInt[unsigned(indexVector[shapeIndex])];

            shapeRef->writeIntAttribute(shapeIndex, shapeRef->getDBFFieldIndex(valFieldOutput.c_str()), varFieldFound);
        }
        else if (fieldType == FTDouble)
        {
            double varFieldFound;
            if (indexVector[shapeIndex] == NODATA)
                varFieldFound = NODATA;
            else
                varFieldFound = varFieldVectorDouble[unsigned(indexVector[shapeIndex])];

            shapeRef->writeDoubleAttribute(shapeIndex, shapeRef->getDBFFieldIndex(valFieldOutput.c_str()), varFieldFound);
        }
    }

    // close and re-open to write also the last shape
    shapeRef->close();
    shapeRef->open(shapeRef->getFilepath());

    return true;
}


