#include <float.h>
#include <algorithm>
#include <math.h>

#include "commonConstants.h"
#include "basicMath.h"
#include "shapeToRaster.h"
#include "zonalStatistic.h"

std::vector <std::vector<int> > computeMatrixAnalysis(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                          gis::Crit3DRasterGrid &rasterRef, gis::Crit3DRasterGrid &rasterVal, std::vector<int> &vectorNull)
{
    unsigned int nrRefShapes = unsigned(shapeRef.getShapeCount());
    unsigned int nrValShapes = unsigned(shapeVal.getShapeCount());

    // analysis matrix
    vectorNull.clear();
    vectorNull.resize(nrRefShapes, 0);
    std::vector <std::vector<int>> matrix(nrRefShapes, std::vector<int>(nrValShapes, 0));

    for (int row = 0; row < rasterRef.header->nrRows; row++)
    {
       for (int col = 0; col < rasterRef.header->nrCols; col++)
       {
           int refIndex = int(rasterRef.value[row][col]);
           if (refIndex != NODATA && refIndex < signed(nrRefShapes))
           {
               double x, y;
               rasterRef.getXY(row, col, x, y);
               if (! gis::isOutOfGridXY(x, y, rasterVal.header))
               {
                    int rowVal, colVal;
                    gis::getRowColFromXY(*(rasterVal.header), x, y, &rowVal, &colVal);

                    int valIndex = int(rasterVal.value[rowVal][colVal]);
                    if (valIndex != NODATA && valIndex < signed(nrValShapes))
                    {
                        matrix[unsigned(refIndex)][unsigned(valIndex)]++;
                    }
                    else
                    {
                        vectorNull[unsigned(refIndex)]++;
                    }
               }
               else
               {
                   vectorNull[unsigned(refIndex)]++;
               }
           }
       }
   }

   return matrix;
}


bool zonalStatisticsShape(Crit3DShapeHandler& shapeRef, Crit3DShapeHandler& shapeVal,
                          std::vector <std::vector<int> > &matrix, std::vector<int> &vectorNull,
                          std::string valField, std::string valFieldOutput, std::string aggregationType,
                          double threshold, std::string& errorStr)
{
    // check if valField exists
    int fieldIndex = shapeVal.getDBFFieldIndex(valField.c_str());
    if (fieldIndex == -1)
    {
        errorStr = shapeVal.getFilepath() + " has not field called " + valField.c_str();
        return false;
    }

    // add new field to shapeRef
    DBFFieldType fieldType = shapeVal.getFieldType(fieldIndex);
    // limit of 10 characters for valFieldOutput
    shapeRef.addField(valFieldOutput.c_str(), fieldType, shapeVal.nWidthField(fieldIndex), shapeVal.nDecimalsField(fieldIndex));

    unsigned int nrRefShapes = unsigned(shapeRef.getShapeCount());
    unsigned int nrValShapes = unsigned(shapeVal.getShapeCount());
    double value = 0;
    double currentValue = 0;
    double sumValues = 0;
    std::vector<int> validPoints(nrRefShapes, 0);
    std::vector<double> aggregationValues(nrRefShapes, NODATA);

    for (unsigned int row = 0; row < nrRefShapes; row++)
    {
        sumValues = 0;
        currentValue = NODATA;

        for (unsigned int col = 0; col < nrValShapes; col++)
        {
            int nrPoints = int(matrix[row][unsigned(col)]);
            if (nrPoints > 0)
            {
                value = shapeVal.getNumericValue(signed(col), fieldIndex);

                if (isEqual(value, NODATA))
                {
                    vectorNull[row] += nrPoints;
                }
                else
                {
                    validPoints[row] += nrPoints;

                    if (int(currentValue) == NODATA)
                    {
                        currentValue = value;
                    }

                    if (aggregationType == "AVG")
                    {
                        sumValues += nrPoints*value;
                    }
                    else if (aggregationType == "MIN")
                    {
                        currentValue = MINVALUE(value, currentValue);
                    }
                    else if (aggregationType == "MAX")
                    {
                        currentValue = MAXVALUE(value, currentValue);
                    }
                }
            }
        }

        // check percentage of valid values
        bool isValid = false;
        if (validPoints[row] > 0)
        {
            double validPercentage = double(validPoints[row]) / double(validPoints[row] + vectorNull[row]);
            if (validPercentage >= threshold)
                isValid = true;
        }

        // aggregation values
        if (! isValid)
        {
            aggregationValues[row] = NODATA;
        }
        else
        {
            if (aggregationType == "AVG")
            {
                aggregationValues[row] = sumValues / validPoints[row];
            }
            else if (aggregationType == "MIN" || aggregationType == "MAX")
            {
                aggregationValues[row] = currentValue;
            }
        }
    }

    // save aggregation values: each row of matrix is a shape of shapeRef
    double valueToSave = 0.0;
    for (unsigned int shapeIndex = 0; shapeIndex < nrRefShapes; shapeIndex++)
    {
        valueToSave = aggregationValues[shapeIndex];
        int fieldIndex = shapeRef.getDBFFieldIndex(valFieldOutput.c_str());
        if (fieldIndex == -1)
        {
            errorStr = "Wrong shape field name: " + valFieldOutput;
            return false;
        }

        if (fieldType == FTInteger)
        {
            shapeRef.writeIntAttribute(int(shapeIndex), fieldIndex, int(valueToSave));
        }
        else if (fieldType == FTDouble)
        {
            shapeRef.writeDoubleAttribute(int(shapeIndex), fieldIndex, valueToSave);
        }
    }

    // close and re-open to write also the last shape
    shapeRef.close();
    shapeRef.open(shapeRef.getFilepath());

    return true;
}


bool zonalStatisticsShapeMajority(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                          std::vector <std::vector<int> >&matrix, std::vector<int> &vectorNull,
                          std::string valField, std::string valFieldOutput,
                          double threshold, std::string &errorStr)
{
    // check if valField exists
    int fieldIndex = shapeVal.getDBFFieldIndex(valField.c_str());
    if (fieldIndex == -1)
    {
        errorStr = shapeVal.getFilepath() + "has not field called " + valField.c_str();
        return false;
    }

    // add new field to shapeRef
    DBFFieldType fieldType = shapeVal.getFieldType(fieldIndex);
    shapeRef.addField(valFieldOutput.c_str(), fieldType, shapeVal.nWidthField(fieldIndex), shapeVal.nDecimalsField(fieldIndex));

    unsigned int nrRefShapes = unsigned(shapeRef.getShapeCount());
    unsigned int nrValShapes = unsigned(shapeVal.getShapeCount());

    std::vector<std::string> vectorValuesString;
    std::vector<double> vectorValuesDouble;
    std::vector<int> vectorValuesInt;
    std::vector<int> vectorNrElements;
    std::vector<int> validPoints(nrRefShapes, 0);

    for (unsigned int row = 0; row < nrRefShapes; row++)
    {
        vectorValuesString.clear();
        vectorValuesDouble.clear();
        vectorValuesInt.clear();
        vectorNrElements.clear();

        for (unsigned int col = 0; col < nrValShapes; col++)
        {
            int nrPoints = int(matrix[row][col]);
            if (nrPoints > 0)
            {
                if (fieldType == FTInteger || fieldType == FTDouble)
                {
                    double value = shapeVal.getNumericValue(int(col), fieldIndex);
                    if (isEqual(value, NODATA))
                    {
                        vectorNull[row] += nrPoints;
                    }
                    else
                    {
                        if (fieldType == FTInteger)
                        {
                            validPoints[row] += nrPoints;
                            std::vector<int>::iterator it;
                            it = std::find (vectorValuesInt.begin(), vectorValuesInt.end(), value);
                            if ( it == vectorValuesInt.end())
                            {
                                // not found - append new value
                                vectorValuesInt.push_back(int(value));
                                vectorNrElements.push_back(nrPoints);
                            }
                            else
                            {
                                unsigned long k = unsigned(it - vectorValuesInt.begin());
                                vectorNrElements[k] += nrPoints;
                            }
                        }
                        else if (fieldType == FTDouble)
                        {
                            validPoints[row] += nrPoints;
                            std::vector<double>::iterator it;
                            it = std::find (vectorValuesDouble.begin(), vectorValuesDouble.end(), value);
                            if ( it == vectorValuesDouble.end())
                            {
                                // not found - append new value
                                vectorValuesDouble.push_back(value);
                                vectorNrElements.push_back(nrPoints);
                            }
                            else
                            {
                                unsigned long k = unsigned(it - vectorValuesDouble.begin());
                                vectorNrElements[k] += nrPoints;
                            }
                        }
                    }
                }
                else if (fieldType == FTString)
                {
                    std::string strValue = shapeVal.readStringAttribute(signed(col),fieldIndex);
                    if (strValue == "" || strValue == "-9999" || strValue == "******")
                    {
                        vectorNull[row] += nrPoints;
                    }
                    else
                    {
                        validPoints[row] += nrPoints;
                        std::vector<std::string>::iterator it;
                        it = std::find (vectorValuesString.begin(), vectorValuesString.end(), strValue);
                        if ( it == vectorValuesString.end())
                        {
                            // not found - append new value
                            vectorValuesString.push_back(strValue);
                            vectorNrElements.push_back(nrPoints);
                        }
                        else
                        {
                            unsigned long k = unsigned(it - vectorValuesString.begin());
                            vectorNrElements[k] += nrPoints;
                        }
                    }
                }
            }
        }

        // check valid values
        bool isValid = false;
        if (validPoints[row] > 0)
        {
            double validPercentage = double(validPoints[row]) / double(validPoints[row] + vectorNull[row]);
            if (validPercentage >= threshold)
                isValid = true;
        }

        if (! isValid)
        {
            // write NODATA or null string
            if (fieldType == FTInteger)
            {
                shapeRef.writeIntAttribute(signed(row), shapeRef.getDBFFieldIndex(valFieldOutput.c_str()), NODATA);
            }
            else if (fieldType == FTDouble)
            {
                shapeRef.writeDoubleAttribute(signed(row), shapeRef.getDBFFieldIndex(valFieldOutput.c_str()), NODATA);
            }
            else if (fieldType == FTString)
            {
                shapeRef.writeStringAttribute(signed(row), shapeRef.getDBFFieldIndex(valFieldOutput.c_str()), "");
            }
        }
        else
        {
            // search index of prevailing value
            int maxValue = 0;
            unsigned int index = 0;
            for (unsigned int i = 0; i < vectorNrElements.size(); i++)
            {
                if (vectorNrElements[i] > maxValue)
                {
                    maxValue = vectorNrElements[i];
                    index = i;
                }
            }

            if (fieldType == FTInteger)
            {
                shapeRef.writeIntAttribute(signed(row), shapeRef.getDBFFieldIndex(valFieldOutput.c_str()), vectorValuesInt[index]);
            }
            else if (fieldType == FTDouble)
            {
                shapeRef.writeDoubleAttribute(signed(row), shapeRef.getDBFFieldIndex(valFieldOutput.c_str()), vectorValuesDouble[index]);
            }
            else if (fieldType == FTString)
            {
                shapeRef.writeStringAttribute(signed(row), shapeRef.getDBFFieldIndex(valFieldOutput.c_str()), vectorValuesString[index].c_str());
            }
        }
    }

    vectorValuesString.clear();
    vectorValuesDouble.clear();
    vectorValuesInt.clear();
    vectorNrElements.clear();
    validPoints.clear();

    // close and re-open to write also the last shape
    shapeRef.close();
    shapeRef.open(shapeRef.getFilepath());

    return true;
}
