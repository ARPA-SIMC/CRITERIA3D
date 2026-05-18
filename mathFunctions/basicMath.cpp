/*!
    \copyright 2023
    Fausto Tomei, Gabriele Antolini, Antonio Volta

    This file is part of AGROLIB distribution.
    AGROLIB has been developed under contract issued by A.R.P.A. Emilia-Romagna

    AGROLIB is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AGROLIB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with AGROLIB.  If not, see <http://www.gnu.org/licenses/>.

    Contacts:
    ftomei@arpae.it
    gantolini@arpae.it
    avolta@arpae.it
*/

#include <math.h>
#include <stdlib.h>
#include <algorithm>

#include "commonConstants.h"
#include "basicMath.h"


    bool sameSignNoZero(float a, float b)
    {
        return a*b > 0.0f;
    }

    bool sameSign(float a, float b)
    {
        return a*b >= 0.0f;
    }

    bool greaterThan(float a, float b)
    {
        return (fabs(a - b) > fabs(b / 100.f));
    }

    bool greaterThan(double a, double b)
    {
        return (fabs(a - b) > fabs(b / 100.));
    }

    bool compareValue(float a, float b, bool isPositive)
    {
        if (isPositive)
        {
            return (a > b);
        }
        else
        {
            return (a < b);
        }
    }

    int integralPart(double number)
    {
        double intPart;
        modf(number,&intPart);
        return int(intPart);
    }

    double fractionalPart(double number)
    {
        double intPart,fracPart;
        fracPart = modf(number,&intPart);
        return fracPart;
    }

    double inputSwitch (double x, double y1, double y2)
    {
        if (x < 0) return y1;
        else return y2;
    }

    double stepFunction (double x, double change, double y1, double y2)
    {
        if (x < change) return y1;
        else return y2;
    }

    double boundedValue (double x, double lowerBound, double upperBound)
    {
        if (x < lowerBound) return lowerBound;
        else if (x > upperBound) return upperBound;
        else return x;
    }

    void directRotation(float *point, float angle)
    {
        point[0] = cosf(angle)*point[0] - sinf(angle)*point[1];
        point[0] = sinf(angle)*point[0] + cosf(angle)*point[1];
    }

    void inverseRotation(float *point, float angle)
    {
        angle *=-1;
        point[0] = cosf(angle)*point[0] - sinf(angle)*point[1];
        point[0] = sinf(angle)*point[0] + cosf(angle)*point[1];
    }

    float distance(float* x,float* y, int vectorLength)
    {
        float dist = 0 ;
        for (int i=0; i<vectorLength;i++)
            dist = powf(x[i]-y[i],2);

        dist = sqrtf(dist);
        return dist;
    }

    float distance2D(float x1, float y1, float x2, float y2)
    {
        return sqrtf((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    }

    float norm(float* x, int vectorLength)
    {
        float myNorm = 0 ;
        for (int i=0; i<vectorLength;i++)
            myNorm = powf(x[i],2);

        myNorm = sqrtf(myNorm);
        return myNorm;
    }

    bool findLinesIntersection(float q1, float m1, float q2, float m2, float* x, float* y)
    {
        if (! isEqual(m1, m2))
        {
            *x = (q2 - q1) / (m1 - m2);
            *y = m1 * (q2 - q1) / (m1 - m2) + q1;
            return true;
        }
        else
        {
            *x = NODATA;
            *y = NODATA;
            return false;
        }
    }

    bool findLinesIntersectionAboveThreshold(float q1, float m1, float q2, float m2, float myThreshold, float* x, float* y)
    {
        if (! isEqual(m1, m2))
        {
            *x = (q2 - q1) / (m1 - m2);
            *y = m1 * (q2 - q1) / (m1 - m2) + q1;
            if (*x > myThreshold)
                return true;
            else
                return false;
        }
        else
        {
            *x = NODATA;
            *y = NODATA;
            return false;
        }
    }

    int sgn(float v)
    {
      if (v < 0) return -1;
      if (v > 0) return 1;
      return 0;
    }

    char* decimal_to_binary(unsigned int n, int nrBits)
    {
       int d, count;
       char *pointer = new char[unsigned(nrBits)];

       count = 0;
       for (short c = short(nrBits-1); c >= 0 ; c--)
       {
          d = n >> c;

          if (d & 1)
             *(pointer+count) = 1 + '0';
          else
             *(pointer+count) = 0 + '0';

          count++;
       }
       *(pointer+count) = '\0';

       return  pointer;
    }

    float getSinDecimalDegree(float angle)
    {
        while (angle > 360) angle -= 360 ;
        while (angle < -360) angle +=360 ;
        return float(sin(double(angle) * DEG_TO_RAD));
    }

    float getCosDecimalDegree(float angle)
    {
        while (angle > 360) angle -= 360 ;
        while (angle < -360) angle +=360 ;
        return float(cos(double(angle) * DEG_TO_RAD));
    }

    double getSinDecimalDegree(double angle)
    {
        while (angle > 360) angle -= 360 ;
        while (angle < -360) angle +=360 ;
        return sin(angle * DEG_TO_RAD);
    }

    double getCosDecimalDegree(double angle)
    {
        while (angle > 360) angle -= 360 ;
        while (angle < -360) angle +=360 ;
        return cos(angle * DEG_TO_RAD);
    }

    double powerIntegerExponent(double base, int exponent)
    {
        if(exponent > 0)
        {
            double result = 1.0;
            while (exponent > 0)
            {
                if (exponent%2 == 1)  // odd exponent
                    result *= base;
                base *= base;  // the base is doubled
                exponent /= 2;
            }
            return result;
        }
        else if (exponent < 0)
            return 1.0 / powerIntegerExponent(base, -exponent);  // negative exponents
        else
            return 1;
    }

    namespace sorting
    {
    void quicksortAscendingInteger(int* x, int first, int last)
        {
            int i = first;
            int j = last;
            int pivot = x[(first + last) / 2];

            while (i <= j)
            {
                while (x[i] < pivot)
                    i++;

                while (x[j] > pivot)
                    j--;

                if (i <= j)
                {
                    std::swap(x[i], x[j]);
                    i++;
                    j--;
                }
            }

            if (first < j)
                quicksortAscendingInteger(x, first, j);

            if (i < last)
                quicksortAscendingInteger(x, i, last);
        }


        void quicksortDescendingInteger(int *x, int first,int last)
        {
            quicksortAscendingInteger(x, first, last);

            for (int i = first; i < (last/2); i++)
                std::swap(x[i], x[last-i]);
        }


        void quicksortAscendingIntegerWithParameters(std::vector<int>& x, std::vector<float>& values,
                                                     int first, int last)
        {
            int i = first;
            int j = last;

            float pivot = values[(first + last) / 2];

            while (i <= j)
            {
                while (values[i] < pivot)
                    i++;

                while (values[j] > pivot)
                    j--;

                if (i <= j)
                {
                    std::swap(values[i], values[j]);
                    std::swap(x[i], x[j]);

                    i++;
                    j--;
                }
            }

            if (first < j)
                quicksortAscendingIntegerWithParameters(x, values, first, j);

            if (i < last)
                quicksortAscendingIntegerWithParameters(x, values, i, last);
        }


        // isSortValues == true: list will be modified
        // isSortValues == false: list must already be sorted ascending and not contain NODATA values
        float percentile(std::vector<float>& list, int& nrList, double percentage, bool isSortValues)
        {
            // check
            if (list.size() < MINIMUM_PERCENTILE_DATA || (percentage <= 0.0) || (percentage > 100.0))
                return NODATA;

            percentage /= 100.f;

            if (isSortValues)
            {
                // remove nodata
                list.erase(std::remove_if(list.begin(), list.end(),
                            [](float v){ return isEqual(v, NODATA); }),
                            list.end());

                // sort
                std::sort(list.begin(), list.end());

                nrList = int(list.size());

                // check on data presence
                if (nrList < MINIMUM_PERCENTILE_DATA)
                    return NODATA;
            }

            double rank = nrList * percentage - 1.0;

            // return percentile
            if (rank < 0.0)
                return list[0];

            int low = static_cast<int>(rank);

            if (low >= (nrList - 1))
                return list[nrList - 1];

            double frac = rank - low;

            return float(list[low] + frac * (list[low + 1] - list[low]));
        }


        // isSortValues == true: list will be modified
        // isSortValues == false: list must already be sorted ascending and not contain NODATA values
        // this implementation uses convention: rank = (i+1)/N
        float percentileRank(std::vector<float>& list, float value, bool isSortValues)
        {
            if (list.size() < MINIMUM_PERCENTILE_DATA)
                return NODATA;

            if (isSortValues)
            {
                // remove nodata
                list.erase(std::remove_if(list.begin(), list.end(),
                            [](float v){ return isEqual(v, NODATA); }),
                            list.end());

                // check on data presence
                if (list.size() < MINIMUM_PERCENTILE_DATA)
                    return NODATA;

                // sort
                std::sort(list.begin(), list.end());
            }

            const int n = static_cast<int>(list.size());
            const int lastIndex = n - 1;

            // return rank
            if (value <= list[0])
                return 0.0;
            if (value >= list[lastIndex])
                return 100.0;

            for (size_t i = 0; i < list.size(); i++)
            {
                double rank = double(i + 1) / double(n);

                if (isEqual(value, list[i]))
                {
                    return float(rank * 100.0);
                }

                if (i < lastIndex && list[i] < value && list[i + 1] > value)
                {
                    double delta = list[i + 1] - list[i];

                    if (std::abs(delta) < 1e-12)
                        return float(rank * 100.0);

                    rank += (value - list[i]) / delta / double(n);

                    return float(rank * 100.0);
                }
            }

            return NODATA;
        }


        // warning: if isSortValues is true, list will be modified
        // if isSortValues == false:
        // - list must already be sorted ascending
        // - list must not contain NODATA
        // - list must contain only values >= threshold
        float percentileAboveThreshold(std::vector<float>& list, int& nrList,
                                       float percentage, float threshold, bool isSortValues)
        {
            nrList = std::min<int>(nrList, static_cast<int>(list.size()));

            // check
            if (nrList < MINIMUM_PERCENTILE_DATA || (percentage <= 0) || (percentage > 100))
                return NODATA;

            percentage /= 100.f;

            if (isSortValues)
            {
                // remove nodata
                list.erase(std::remove_if(list.begin(), list.end(),
                            [](float v){ return isEqual(v, NODATA); }),
                            list.end());

                // remove under threshold
                list.erase(std::remove_if(list.begin(), list.end(), [threshold](const float& x)
                                          {return x < threshold; }), list.end());

                // sort
                std::sort(list.begin(), list.end());

                nrList = int(list.size());

                // check on data presence
                if (nrList < MINIMUM_PERCENTILE_DATA)
                    return NODATA;
            }

            float rank = nrList * percentage - 1.f;

            if (rank < 0.f)
                return list[0];

            int low = static_cast<int>(rank);

            if (low >= nrList - 1)
                return list[nrList - 1];

            float frac = rank - low;

            return list[low] + frac * (list[low + 1] - list[low]);
        }


        // warning: if isSortValues is true, list will be modified
        float mode(std::vector<float> &list, int* nrList, bool isSortValues)
        {
            if (list.empty() || *nrList <= 0)
                return NODATA;

            if (isSortValues)
            {
                // clean nodata
                std::vector<float> cleanList;

                for (int i = 0; i < *nrList; i++)
                {
                    if (! isEqual(list[i], NODATA))
                        cleanList.push_back(list[i]);
                }

                if (cleanList.empty())
                {
                    *nrList = 0;
                    return NODATA;
                }

                std::sort(cleanList.begin(), cleanList.end());

                list = cleanList;
                *nrList = static_cast<int>(cleanList.size());
            }

            // find max frequency
            int max_count = 1;
            int count = 1;
            float res = list[0];

            for (int i = 1; i < *nrList; i++)
            {
                if (isEqual(list[i], list[i - 1]))
                {
                    count++;
                }
                else
                {
                    if (count > max_count)
                    {
                        max_count = count;
                        res = list[i - 1];
                    }
                    count = 1;
                }
            }

            // when the last element is most frequent
            if (count > max_count)
            {
                res = list[*nrList - 1];
            }

            return res;
        }
    }

