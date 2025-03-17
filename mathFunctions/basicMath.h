#ifndef BASICMATH_H
#define BASICMATH_H

    #ifndef _VECTOR_
        #include <vector>
    #endif
    #ifndef POWER4
        #define POWER4(x) ((x) * (x) * (x) * (x))
    #endif
    #ifndef POWER3
        #define POWER3(x) ((x) * (x) * (x))
    #endif
    #ifndef POWER2
        #define POWER2(x) ((x) * (x))
    #endif
    #ifndef LOGICAL_IO
        #define LOGICAL_IO(logicCondition, val1, val2) ((logicCondition) ? (val1) : (val2))
    #endif
    bool sameSignNoZero(float a, float b);
    bool sameSign(float a, float b);
    bool greaterThan(float a, float b);
    bool greaterThan(double a, double b);
    bool compareValue(float a, float b, bool isPositive);
    int integralPart(double number);
    double fractionalPart(double number);
    double inputSwitch (double x, double y1, double y2);
    double stepFunction (double x, double change, double y1, double y2);
    double boundedValue (double x, double lowerBound, double upperBound);
    void directRotation(float *point, float angle);
    void inverseRotation(float *point, float angle);
    float distance(float* x,float* y, int vectorLength);
    float distance2D(float x1,float y1, float x2, float y2);
    float norm(float* x, int vectorLength);
    bool findLinesIntersection(float q1, float m1, float q2, float m2, float* x, float* y);
    bool findLinesIntersectionAboveThreshold(float q1, float m1, float q2, float m2, float myThreshold, float* x, float* y);
    int sgn(float v);
    bool isEqual(float value1, float value2);
    bool isEqual(double value1, double value2);
    char* decimal_to_binary(unsigned int n, int nrBits);
    float getSinDecimalDegree(float angle);
    float getCosDecimalDegree(float angle);
    double powerIntegerExponent(double base, int exponent);

    namespace sorting
    {
        void quicksortAscendingInteger(int *x,int first, int last);
        void quicksortDescendingInteger(int *x, int first,int last);
        void quicksortAscendingIntegerWithParameters(std::vector<int> &x, std::vector<float> &values, unsigned first, unsigned last);
        void quicksortAscendingDouble(double *x, int first,int last);
        void quicksortAscendingFloat(std::vector<float> &values, unsigned int first, unsigned int last);
        float percentile(std::vector<float>& list, int& nrList, float perc, bool sortValues);
        float percentileRank(std::vector<float> &list, float value, bool isSortValues);
        float mode(std::vector<float> &list, int* nrList, bool isSortValues);
    }


#endif // BASICMATH_H
