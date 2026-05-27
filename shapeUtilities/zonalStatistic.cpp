#include <float.h>
#include <algorithm>
#include <math.h>
#include <unordered_map>

#include "commonConstants.h"
#include "basicMath.h"
#include "shapeToRaster.h"
#include "zonalStatistic.h"
#include "gis.h"
#include "shapeHandler.h"


std::vector <std::vector<int>> computeMatrixAnalysis(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                          gis::Crit3DRasterGrid &rasterRef, gis::Crit3DRasterGrid &rasterVal, std::vector<int> &vectorNull)
{
    unsigned int nrRefShapes = unsigned(shapeRef.getShapeCount());
    unsigned int nrValShapes = unsigned(shapeVal.getShapeCount());

    // analysis matrix
    vectorNull.clear();
    vectorNull.resize(nrRefShapes, 0);
    std::vector <std::vector<int>> matrix(nrRefShapes, std::vector<int>(nrValShapes, 0));

    double x, y;
    int rowVal, colVal;
    int flagRef = int(rasterRef.header->flag);
    int flagVal = int(rasterVal.header->flag);
    for (int row = 0; row < rasterRef.header->nrRows; row++)
    {
        for (int col = 0; col < rasterRef.header->nrCols; col++)
        {
            int refIndex = int(rasterRef.value[row][col]);
            if (refIndex != flagRef && refIndex >= 0 && refIndex < signed(nrRefShapes))
            {
                rasterRef.getXY(row, col, x, y);
                if (gis::isOutOfGridXY(x, y, rasterVal.header))
                {
                    vectorNull[unsigned(refIndex)]++;
                    continue;
                }

                gis::getRowColFromXY(*(rasterVal.header), x, y, rowVal, colVal);

                int valIndex = int(rasterVal.value[rowVal][colVal]);
                if (valIndex != flagVal && valIndex >= 0 && valIndex < signed(nrValShapes))
                {
                    matrix[unsigned(refIndex)][unsigned(valIndex)]++;
                }
                else
                    vectorNull[unsigned(refIndex)]++;
            }
        }
    }

   return matrix;
}


std::vector <std::vector<int>> computeMatrixAnalysisRaster(const Crit3DShapeHandler &shapeRef, const gis::Crit3DRasterGrid &rasterVal,
                                                           std::vector<int> &categories, std::vector<int> &vectorNull)
{
    unsigned int nrRefShapes = unsigned(shapeRef.getShapeCount());

    // extract categories from rasterVal
    categories.clear();
    categories = gis::extractUniqueValues(rasterVal);
    size_t nrCategories = categories.size();

    // unordered map is faster for search
    std::unordered_map<int, int> categoryIndex;
    for (unsigned int i = 0; i < nrCategories; ++i) {
        categoryIndex[categories[i]] = i;
    }

    // create reference raster from shapefile (same header of rasterVal)
    gis::Crit3DRasterGrid rasterRef;
    rasterRef.initializeGrid(*(rasterVal.header));
    fillRasterWithShapeNumber(rasterRef, shapeRef);

    // analysis matrix
    vectorNull.clear();
    vectorNull.resize(nrRefShapes, 0);
    std::vector <std::vector<int>> matrix(nrRefShapes, std::vector<int>(nrCategories, 0));

    int flagInt = int(rasterRef.header->flag);
    for (int row = 0; row < rasterRef.header->nrRows; row++)
    {
        for (int col = 0; col < rasterRef.header->nrCols; col++)
        {
            int refIndex = int(rasterRef.value[row][col]);
            if (refIndex != flagInt && refIndex >= 0 && refIndex < static_cast<int>(nrRefShapes))
            {
                int valueInt = int(rasterVal.value[row][col]);
                if (valueInt == flagInt)
                {
                    vectorNull[refIndex]++;
                    continue;
                }

                auto it = categoryIndex.find(valueInt);

                if (it != categoryIndex.end())
                {
                    int valIndex = it->second;
                    matrix[refIndex][valIndex]++;
                }
                else
                {
                    vectorNull[refIndex]++;
                }
            }
        }
    }

    return matrix;
}


bool zonalStatisticsShape(Crit3DShapeHandler& shapeRef, Crit3DShapeHandler& shapeVal,
                          const std::vector <std::vector<int>> &matrix, std::vector<int> &vectorNull,
                          const std::string &valField, const std::string &valFieldOutput,
                          const std::string &aggregationType, double threshold, std::string& errorStr)
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

    return true;
}


bool zonalStatisticsShapeMajority_old(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                          const std::vector <std::vector<int>> &matrix, std::vector<int> &vectorNull,
                          const std::string &valField, const std::string &fieldOutput,
                          double threshold, std::string &errorStr)
{
    // check if valField exists
    int fieldIndex = shapeVal.getDBFFieldIndex(valField.c_str());
    if (fieldIndex == -1)
    {
        errorStr = shapeVal.getFilepath() + "has not field called " + valField.c_str();
        return false;
    }
    DBFFieldType fieldType = shapeVal.getFieldType(fieldIndex);

    // add new field to shapeRef
    if (! shapeRef.existField(fieldOutput))
    {
        if (! shapeRef.addField(fieldOutput.c_str(), fieldType, shapeVal.nWidthField(fieldIndex),
                               shapeVal.nDecimalsField(fieldIndex)) )
        {
            errorStr = "error writing new field: " + fieldOutput;
            return false;
        }
    }

    int fieldOutputIndex = shapeRef.getDBFFieldIndex(fieldOutput.c_str());

    unsigned int nrRefShapes = unsigned(shapeRef.getShapeCount());
    unsigned int nrValShapes = unsigned(shapeVal.getShapeCount());

    std::vector<std::string> vectorValuesString;
    std::vector<double> vectorValuesDouble;
    std::vector<int> vectorValuesInt;
    std::vector<int> vectorNrElements;

    for (unsigned int row = 0; row < nrRefShapes; row++)
    {
        vectorValuesString.clear();
        vectorValuesDouble.clear();
        vectorValuesInt.clear();
        vectorNrElements.clear();
        int validPoints = 0;

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
                            validPoints += nrPoints;
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
                            validPoints += nrPoints;
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
                        validPoints += nrPoints;
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

        // check validity
        bool isValid = false;
        if ((validPoints + vectorNull[row]) > 0)
        {
            double validPercentage = double(validPoints) / double(validPoints + vectorNull[row]);
            isValid = (validPercentage >= threshold);
        }

        if (! isValid)
        {
            // write NODATA or null string
            if (fieldType == FTInteger)
            {
                shapeRef.writeIntAttribute(signed(row), fieldOutputIndex, NODATA);
            }
            else if (fieldType == FTDouble)
            {
                shapeRef.writeDoubleAttribute(signed(row), fieldOutputIndex, NODATA);
            }
            else if (fieldType == FTString)
            {
                shapeRef.writeStringAttribute(signed(row), fieldOutputIndex, "");
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
                shapeRef.writeIntAttribute(signed(row), fieldOutputIndex, vectorValuesInt[index]);
            }
            else if (fieldType == FTDouble)
            {
                shapeRef.writeDoubleAttribute(signed(row), fieldOutputIndex, vectorValuesDouble[index]);
            }
            else if (fieldType == FTString)
            {
                shapeRef.writeStringAttribute(signed(row), fieldOutputIndex, vectorValuesString[index].c_str());
            }
        }
    }

    vectorValuesString.clear();
    vectorValuesDouble.clear();
    vectorValuesInt.clear();
    vectorNrElements.clear();

    // close and re-open to write also the last shape
    shapeRef.close();
    shapeRef.open(shapeRef.getFilepath(), false);

    return true;
}


bool zonalStatisticsShapeMajority( Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
        const std::vector<std::vector<int>> &matrix, std::vector<int> &vectorNull,
        const std::string &valField, const std::string &fieldOutput,
        double threshold, std::string &errorStr)
{
    // ----------------------------
    // CHECK INPUT FIELD
    // ----------------------------
    int fieldIndex = shapeVal.getDBFFieldIndex(valField.c_str());
    if (fieldIndex == -1)
    {
        errorStr = shapeVal.getFilepath() + " has not field called " + valField;
        return false;
    }

    DBFFieldType fieldType = shapeVal.getFieldType(fieldIndex);
    bool isNumeric = (fieldType == FTInteger || fieldType == FTDouble);

    // ----------------------------
    // CREATE OUTPUT FIELD
    // ----------------------------
    if (!shapeRef.existField(fieldOutput))
    {
        if (!shapeRef.addField(fieldOutput.c_str(),
                               fieldType,
                               shapeVal.nWidthField(fieldIndex),
                               shapeVal.nDecimalsField(fieldIndex)))
        {
            errorStr = "error writing new field: " + fieldOutput;
            return false;
        }
    }

    int fieldOutputIndex = shapeRef.getDBFFieldIndex(fieldOutput.c_str());

    const unsigned int nrRefShapes = shapeRef.getShapeCount();
    const unsigned int nrValShapes = shapeVal.getShapeCount();

    for (unsigned int row = 0; row < nrRefShapes; ++row)
    {
        long validPoints = 0;
        std::unordered_map<std::string, long> freqStr;
        std::unordered_map<int, long> freqInt;
        std::unordered_map<double, long> freqDouble;

        for (unsigned int col = 0; col < nrValShapes; ++col)
        {
            int nrPoints = matrix[row][col];
            if (nrPoints <= 0) continue;

            if (isNumeric)
            {
                double value = shapeVal.getNumericValue(col, fieldIndex);

                if (isEqual(value, NODATA))
                {
                    vectorNull[row] += nrPoints;
                    continue;
                }

                validPoints += nrPoints;

                if (fieldType == FTInteger)
                {
                    freqInt[(int)value] += nrPoints;
                }
                else
                {
                    freqDouble[value] += nrPoints;
                }
            }
            else // STRING
            {
                std::string value = shapeVal.readStringAttribute(col, fieldIndex);

                if (value.empty() || value == "-9999" || value == "******")
                {
                    vectorNull[row] += nrPoints;
                    continue;
                }

                validPoints += nrPoints;
                freqStr[value] += nrPoints;
            }
        }

        // ----------------------------
        // VALIDITY CHECK
        // ----------------------------
        bool isValid = false;

        if ((validPoints + vectorNull[row]) > 0)
        {
            double perc = double(validPoints) /
                          double(validPoints + vectorNull[row]);

            isValid = (perc >= threshold);
        }

        if (! isValid)
        {
            if (fieldType == FTInteger)
                shapeRef.writeIntAttribute(row, fieldOutputIndex, NODATA);
            else if (fieldType == FTDouble)
                shapeRef.writeDoubleAttribute(row, fieldOutputIndex, NODATA);
            else
                shapeRef.writeStringAttribute(row, fieldOutputIndex, "");

            continue;
        }

        // ----------------------------
        // FIND MAJORITY
        // ----------------------------
        long maxCount = 0;

        if (fieldType == FTInteger)
        {
            int majority = NODATA;

            for (const auto &p : freqInt)
            {
                if (p.second > maxCount)
                {
                    maxCount = p.second;
                    majority = p.first;
                }
            }

            shapeRef.writeIntAttribute(row, fieldOutputIndex, majority);
        }
        else if (fieldType == FTDouble)
        {
            double majority = NODATA;

            for (const auto &p : freqDouble)
            {
                if (p.second > maxCount)
                {
                    maxCount = p.second;
                    majority = p.first;
                }
            }

            shapeRef.writeDoubleAttribute(row, fieldOutputIndex, majority);
        }
        else // STRING
        {
            std::string majority = "";

            for (const auto &p : freqStr)
            {
                if (p.second > maxCount)
                {
                    maxCount = p.second;
                    majority = p.first;
                }
            }

            shapeRef.writeStringAttribute(row, fieldOutputIndex, majority.c_str());
        }
    }

    // close and re-open to write also the last shape
    shapeRef.close();
    shapeRef.open(shapeRef.getFilepath(), false);

    return true;
}


// assign values to shape proportional to raster distribution (Hare-Niemeyer method)
// warning: the categories must be integer values
bool zonalStatisticsShapeCategories_proportional(Crit3DShapeHandler &shapeRef, const std::vector<int> &categories,
                                            const std::vector <std::vector<int>> &matrix, double cellSize,
                                            const std::string &fieldName, double threshold, std::string &errorStr)
{
    const unsigned int nrRefShapes = shapeRef.getShapeCount();
    const size_t nrCategories = categories.size();

    std::vector<bool> isShapeValid(nrRefShapes, false);
    std::vector<long> nrOfVotes(nrCategories, 0);

    long nrValidVotes = 0;
    int nrValidSeats = 0;
    double pixelArea = cellSize * cellSize;

    for (unsigned int row = 0; row < nrRefShapes; row++)
    {
        const auto& rowMatrix = matrix[row];
        long nrCurrentValues = 0;

        for (unsigned int col = 0; col < nrCategories; ++col)
        {
            int count = rowMatrix[col];

            if (count > 0)
                nrCurrentValues += count;
        }

        if (nrCurrentValues > 0)
        {
            ShapeObject currentShape;
            if (! shapeRef.getShape(row, currentShape))
            {
                errorStr = "wrong shape.";
                return false;
            }

            double totalArea = currentShape.getTotalArea();
            if (isEqual(totalArea, 0.0))
                continue;

            double validAreaPercentage = (nrCurrentValues * pixelArea) / totalArea;

            if (validAreaPercentage >= threshold)
            {
                isShapeValid[row] = true;
                nrValidSeats++;

                nrValidVotes += nrCurrentValues;

                for (unsigned int col = 0; col < nrCategories; ++col)
                    nrOfVotes[col] += rowMatrix[col];
            }
        }
    }

    // check nr of seats
    if (nrValidSeats == 0 || nrValidVotes == 0)
    {
        errorStr = "Not enough valid data.";
        return false;
    }

    std::vector<int> nrOfSeats(nrCategories, 0);
    std::vector<double> remains(nrCategories, 0);
    double nrOfVotesPerSeat = double(nrValidVotes) / double(nrValidSeats);
    int nrOfSeatsAssigned = 0;

    for (unsigned int col = 0; col < nrCategories; ++col)
    {
        nrOfSeats[col] = int(std::floor(double(nrOfVotes[col]) / nrOfVotesPerSeat));
        nrOfSeatsAssigned += nrOfSeats[col];
        remains[col] = nrOfVotes[col] - nrOfSeats[col] * nrOfVotesPerSeat;
    }

    while (nrOfSeatsAssigned < nrValidSeats)
    {
        double maxRemain = 0.0;
        int indexMax = NODATA;

        for (unsigned int col = 0; col < nrCategories; ++col)
        {
            if (remains[col] > maxRemain)
            {
                maxRemain = remains[col];
                indexMax = col;
            }
        }

        if (indexMax == NODATA)
            break;

        nrOfSeatsAssigned++;
        nrOfSeats[indexMax]++;
        remains[indexMax] = 0.0;
    }

    // check nr of seats
    if (nrOfSeatsAssigned != nrValidSeats)
    {
        errorStr = "error in assign value.";
        return false;
    }

    struct SeatVote
    {
        int row;
        int col;
        int votes;
    };

    // collect votes
    std::vector<SeatVote> allVotes;
    allVotes.reserve(nrRefShapes  * nrCategories);

    for (size_t row = 0; row < nrRefShapes; ++row)
    {
        if (! isShapeValid[row])
            continue;

        for (size_t col = 0; col < nrCategories; ++col)
        {
            allVotes.push_back({
                static_cast<int>(row),
                static_cast<int>(col),
                matrix[row][col]
            });
        }
    }

    // sort votes descending
    std::sort(allVotes.begin(), allVotes.end(),
          [](const SeatVote& a, const SeatVote& b)
          {
              return a.votes > b.votes;
          });

    struct CategoryPriority
    {
        int col;
        int seats;
        long totalVotes;
    };

    // collect priorities
    std::vector<CategoryPriority> categoryPriority;

    for (size_t col = 0; col < nrCategories; ++col)
    {
        if (nrOfSeats[col] > 0)
        {
            categoryPriority.push_back({
                static_cast<int>(col),
                nrOfSeats[col],
                nrOfVotes[col]
            });
        }
    }

    // sort priorities: weaker categories first
    std::sort(categoryPriority.begin(), categoryPriority.end(),
          [](const auto& a, const auto& b)
          {
              if (a.seats != b.seats)
                  return a.seats < b.seats;
              return a.totalVotes < b.totalVotes;
          });

    // split votes by category
    std::vector<std::vector<SeatVote>> categoryVotes(nrCategories);
    for (const auto& vote : allVotes)
    {
        categoryVotes[vote.col].push_back(vote);
    }

    // ----------------------------------------
    // assignment (round robin)
    // ----------------------------------------
    std::vector<int> assignedCategory(nrRefShapes, NODATA);
    std::vector<int> assignedSeats(nrCategories, 0);

    // current scanning index for each category
    std::vector<size_t> currentIndex(nrCategories, 0);

    long totalAssignedSeats = 0;
    bool assignedSomething = true;

    while (assignedSomething && totalAssignedSeats < nrValidSeats)
    {
        assignedSomething = false;

        for (const auto& cat : categoryPriority)
        {
            const int col = cat.col;

            // category already full
            if (assignedSeats[col] >= nrOfSeats[col])
                continue;

            auto& votes = categoryVotes[col];
            size_t& index = currentIndex[col];

            // search next valid shape
            while (index < votes.size())
            {
                const SeatVote& vote = votes[index];
                ++index;

                // shape already assigned
                if (assignedCategory[vote.row] != NODATA)
                    continue;

                // assign value
                assignedCategory[vote.row] = categories[col];

                assignedSeats[col]++;
                totalAssignedSeats++;

                assignedSomething = true;
                break;
            }
        }
    }

    // check
    if (totalAssignedSeats < nrValidSeats)
    {
        errorStr = "Error assigning some valid shapes.";
        return false;
    }

    if (! shapeRef.existField(fieldName))
    {
        // if it does not exist, add a new (integer) field to shapeRef
        if (! shapeRef.addField(fieldName.c_str(), FTInteger, 6, 0))
        {
            errorStr = "error writing new field: " + fieldName;
            return false;
        }
    }

    int fieldIndex = shapeRef.getDBFFieldIndex(fieldName.c_str());

    for (unsigned int row = 0; row < nrRefShapes; ++row)
    {
        shapeRef.writeIntAttribute(row, fieldIndex, assignedCategory[row]);
    }

    return true;
}


// warning: categories must be integer values
bool zonalStatisticsShapeCategories_majority(Crit3DShapeHandler &shapeRef, const std::vector<int> &categories,
                                  const std::vector <std::vector<int>> &matrix, double cellSize,
                                  const std::string &fieldName, double threshold, std::string &errorStr)
{
    if (! shapeRef.existField(fieldName))
    {
        // add new integer field to shapeRef
        if (! shapeRef.addField(fieldName.c_str(), FTInteger, 6, 0))
        {
            errorStr = "error writing new field: " + fieldName;
            return false;
        }
    }

    int fieldIndex = shapeRef.getDBFFieldIndex(fieldName.c_str());

    const unsigned int nrRefShapes = shapeRef.getShapeCount();
    const size_t nrCategories = categories.size();
    double pixelArea = cellSize * cellSize;

    for (unsigned int row = 0; row < nrRefShapes; row++)
    {
        long nrCurrentValues = 0;
        int maxCount = 0;
        int majorityValue = NODATA;
        const auto& rowMatrix = matrix[row];

        for (unsigned int col = 0; col < nrCategories; ++col)
        {
            int count = rowMatrix[col];

            if (count > 0)
            {
                nrCurrentValues += count;

                if (count > maxCount)
                {
                    maxCount = count;
                    majorityValue = categories[col];
                }
            }
        }

        // check validity
        bool isValid = false;

        if (nrCurrentValues > 0)
        {
            ShapeObject currentShape;

            if (! shapeRef.getShape(row, currentShape))
            {
                errorStr = "Wrong shape.";
                return false;
            }

            double totalArea = currentShape.getTotalArea();
            if (isEqual(totalArea, 0.0))
                continue;

            double validAreaPercentage = (nrCurrentValues * pixelArea) / totalArea;

            isValid = (validAreaPercentage >= threshold);
        }

        if (isValid)
        {
            shapeRef.writeIntAttribute(row, fieldIndex, majorityValue);
        }
        else
        {
            shapeRef.writeIntAttribute(row, fieldIndex, NODATA);
        }
    }

    return true;
}
