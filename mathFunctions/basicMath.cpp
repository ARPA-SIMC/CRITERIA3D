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

    bool isEqual(float value1, float value2)
    {
        return (fabs(double(value1 - value2)) < EPSILON);
    }

    bool isEqual(double value1, double value2)
    {
        return (fabs(value1 - value2) < EPSILON);
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
        void quicksortAscendingInteger(int *x, int first,int last)
        {
           int pivot,j,temp,i;

           if(first<last)
           {
                pivot=first;
                i=first;
                j=last;

                while(i<j){
                    while(i<last && x[i] <= x[pivot])
                        i++;
                    while(x[j]>x[pivot])
                        j--;
                    if(i<j){
                        temp=x[i];
                         x[i]=x[j];
                         x[j]=temp;
                    }
                }
                temp=x[pivot];
                x[pivot]=x[j];
                x[j]=temp;
                quicksortAscendingInteger(x,first,j-1);
                quicksortAscendingInteger(x,j+1,last);
            }
        }

        void quicksortAscendingIntegerWithParameters(std::vector<int> &x, std::vector<float> &values, unsigned first, unsigned last)
        {
           int tmpIndex;
           unsigned l, r;
           float tmpVal, pivot;

           if(first<last)
           {
               // only 2 elements
               if (last-first == 1)
               {
                   if (values[first] > values[last])
                   {
                       //swap
                       tmpIndex = x[last];
                       tmpVal = values[last];
                       values[last] = values[first];
                       values[first] = tmpVal;
                       x[last]= x[first];
                       x[first] = tmpIndex;
                       return;
                   }
               }
               unsigned posPivot = (last - first) / 2 + first;
               pivot = values[posPivot];
               if (values[last] < pivot)
               {
                   //swap
                   tmpIndex = x[last];
                   tmpVal = values[last];
                   values[last] = values[posPivot];
                   values[posPivot] = tmpVal;
                   x[last]= x[posPivot];
                   x[posPivot] = tmpIndex;
               }
               l=first;
               r=last;

               while(l<r)
               {
                   if (values[l] < pivot)
                   {
                         l = l + 1;
                   }
                   else if (values[r] >= pivot)
                   {
                       r = r -1;
                   }
                   else
                   {
                       //swap
                       tmpIndex = x[r];
                       tmpVal = values[r];
                       values[r] = values[l];
                       values[l] = tmpVal;
                       x[r]= x[l];
                       x[l] = tmpIndex;
                   }
               }
               if (l > first)
               {
                   l = l - 1;
               }
               else
               {
                   //swap
                   tmpIndex = x[posPivot];
                   tmpVal = values[posPivot];
                   values[posPivot] = values[first];
                   values[first] = tmpVal;
                   x[posPivot]= x[first];
                   x[first] = tmpIndex;

                   r = r + 1;
               }

               quicksortAscendingIntegerWithParameters(x,values,first,l);
               quicksortAscendingIntegerWithParameters(x,values,r,last);
            }
        }

        void quicksortAscendingDouble(double *x, int first,int last)
        {
           int pivot,j,i;
            double temp;

            if(first<last)
            {
                pivot=first;
                i=first;
                j=last;

                while(i<j){
                    while(i<last && x[i]<=x[pivot])
                        i++;
                    while(x[j]>x[pivot])
                        j--;
                    if(i<j){
                        temp=x[i];
                         x[i]=x[j];
                         x[j]=temp;
                    }
                }
                temp=x[pivot];
                x[pivot]=x[j];
                x[j]=temp;
                quicksortAscendingDouble(x,first,j-1);
                quicksortAscendingDouble(x,j+1,last);
            }
        }


        void quicksortAscendingFloat(std::vector<float> &values, unsigned int first, unsigned int last)
        {
            unsigned int pivot,j,i;
            float temp;

            if(first<last)
            {
                pivot=first;
                i=first;
                j=last;

                while(i<j)
                {
                    while(i<last && values[i] <= values[pivot]) i++;
                    while(values[j] > values[pivot]) j--;
                    if(i<j)
                    {
                        temp = values[i];
                        values[i] = values[j];
                        values[j] = temp;
                    }
                }
                temp = values[pivot];
                values[pivot] = values[j];
                values[j] = temp;
                if (j > first)
                    quicksortAscendingFloat(values, first, j-1);
                if (j < last)
                    quicksortAscendingFloat(values, j+1, last);
            }
        }


        void quicksortDescendingInteger(int *x, int first,int last)
        {
            int temp;
            quicksortAscendingInteger(x,first,last);
            //temp = x[first];
            for (int i = first ; i < (last/2) ; i++)
            {
                //swap
                temp = x[i];
                x[i]= x[last-i];
                x[last-i] = temp;
            }
        }


        // warning: if isSortValues is true, list will be modified
        float percentile(std::vector<float>& list, int& nrList, float perc, bool isSortValues)
        {
            // check
            if (nrList < MINIMUM_PERCENTILE_DATA || (perc <= 0) || (perc > 100)) return NODATA;
            perc /= 100.f;

            if (isSortValues)
            {
                // remove nodata
                list.erase(std::remove(list.begin(), list.end(), float(NODATA)), list.end());

                // sort
                std::sort(list.begin(), list.end());

                nrList = int(list.size());
                // check on data presence
                if (nrList < MINIMUM_PERCENTILE_DATA) return NODATA;
            }

            float rank = float(nrList) * perc -1;

            // return percentile
            if ((int(rank) + 1) > (nrList - 1))
                return list[unsigned(nrList - 1)];
            else if (rank < 0)
                return list[0];
            else
                return ((rank - int(rank)) * (list[unsigned(rank) + 1] - list[unsigned(rank)])) + list[unsigned(rank)];
        }


        // warning: if isSortValues is true, list will be modified
        float percentileRank(std::vector<float>& list, float value, bool isSortValues)
        {
            if (isSortValues)
            {
                // remove nodata
                list.erase(std::remove(list.begin(), list.end(), float(NODATA)), list.end());

                // check on data presence
                if (list.size() < MINIMUM_PERCENTILE_DATA) return NODATA;

                // sort
                std::sort(list.begin(), list.end());
            }

            float nrValuesF = float(list.size());
            unsigned int lastIndex = unsigned(list.size() - 1);

            // return rank
            if (value <= list[0]) return 0;
            if (value >= list[lastIndex]) return 100;

            for (unsigned int i = 0; i < list.size(); i++)
            {
                if (isEqual(value, list[i]))
                {
                    float rank = float(i + 1) / nrValuesF;
                    return rank * 100.f;
                }
                if (i < lastIndex && list[i] < value && list[i+1] > value)
                {
                    float rank = float(i + 1) / nrValuesF;
                    rank += (value - list[i]) / (list[i+1] - list[i]) / nrValuesF;
                    return rank * 100.f;
                }
            }

            return NODATA;
        }


        // warning: if isSortValues is true, list will be modified
        float mode(std::vector<float> &list, int* nrList, bool isSortValues)
        {

            if (isSortValues)
            {
                // clean nodata
                std::vector<float> cleanList;
                for (unsigned int i = 0; i < unsigned(*nrList); i++)
                {
                    if (int(list[i]) != int(NODATA))
                    {
                        cleanList.push_back(list[i]);
                    }
                }

                // sort
                quicksortAscendingFloat(cleanList, 0, unsigned(cleanList.size() - 1));

                // switch
                *nrList = int(cleanList.size());
                list.clear();
                list = cleanList;
            }

            //finding max frequency
            int max_count = 1;
            float res = list[0];
            int count = 1;
            for (unsigned int i = 1; i < unsigned(*nrList); i++)
            {
                if (isEqual(list[i], list[i - 1]))
                    count++;
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
                max_count = count;
                res = list[unsigned(*nrList) - 1];
            }

            return res;
        }

    }

