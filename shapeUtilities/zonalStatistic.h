#ifndef ZONALSTATISTIC_H
#define ZONALSTATISTIC_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    std::vector <std::vector<int>> computeMatrixAnalysis(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                              gis::Crit3DRasterGrid &rasterRef, gis::Crit3DRasterGrid &rasterVal, std::vector<int> &vectorNull);

    std::vector <std::vector<int>> computeMatrixAnalysisRaster(const Crit3DShapeHandler &shapeRef, const gis::Crit3DRasterGrid &rasterVal,
                                                              std::vector<int> &categories, std::vector<int> &vectorNull);

    bool zonalStatisticsShape(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                              const std::vector<std::vector<int>> &matrix, std::vector<int> &vectorNull,
                              const std::string &valField, const std::string &valFieldOutput,
                              const std::string &aggregationType, double threshold, std::string &errorStr);

    bool zonalStatisticsShapeMajority(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                                      const std::vector <std::vector<int>> &matrix, std::vector<int> &vectorNull,
                                      const std::string &valField, const std::string &fieldOutput,
                                      double threshold, std::string &errorStr);

    bool zonalStatisticsShapeMajorityCategories(Crit3DShapeHandler &shapeRef, const std::vector<int> &categories,
                                                const std::vector <std::vector<int>> &matrix, std::vector<int> &vectorNull,
                                                const std::string &fieldName, double threshold, std::string &errorStr);

#endif // ZONALSTATISTIC_H
