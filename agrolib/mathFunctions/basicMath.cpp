/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    Contacts:
    Antonio Volta  avolta@arpae.it
*/

#include "commonConstants.h"
#include "basicMath.h"
#include <math.h>
#include <stdlib.h>


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

    float distance2D(float x1,float y1, float x2, float y2)
    {
        float dist;
        dist = sqrtf(powf((x1-x2),2) + powf((y1-y2),2));
        return dist;
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
        double percentile(double* list, int* nrList, double perc, bool isSortValues)
        {
            // check
            if (*nrList < MINIMUM_PERCENTILE_DATA || perc <= 0 || perc >= 100) return NODATA;

            perc /= 100.;

            if (isSortValues)
            {
                // clean missing data
                double* cleanList = new double[unsigned(*nrList)];
                int n = 0;
                for (int i = 0; i < *nrList; i++)
                    if (int(list[i]) != int(NODATA))
                        cleanList[n++] = list[i];

                // switch
                *nrList = n;
                *list = *cleanList;
                delete[] cleanList;

                // check on data presence
                if (*nrList < MINIMUM_PERCENTILE_DATA)
                    return NODATA;

                // sort
                quicksortAscendingDouble(list, 0, *nrList - 1);
            }

            double rank = double(*nrList) * perc - 1;

            // return percentile
            if ((rank + 1) > (*nrList - 1))
                return list[*nrList - 1];
            else if (rank < 0)
                return list[0];
            else
                return ((rank - int(rank)) * (list[int(rank) + 1] - list[int(rank)])) + list[int(rank)];
        }


        // warning: if isSortValues is true, list will be modified
        float percentile(std::vector<float> &list, int* nrList, float perc, bool isSortValues)
        {
            // check
            if (*nrList < MINIMUM_PERCENTILE_DATA || perc <= 0 || perc >= 100) return NODATA;

            perc /= 100.f;

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

                // check on data presence
                if (cleanList.size() < MINIMUM_PERCENTILE_DATA)
                    return NODATA;

                // sort
                quicksortAscendingFloat(cleanList, 0, unsigned(cleanList.size() - 1));

                // switch
                *nrList = int(cleanList.size());
                list.clear();
                list = cleanList;
            }

            float rank = float(*nrList) * perc - 1;

            // return percentile
            if ((rank + 1) > (*nrList - 1))
                return list[unsigned(*nrList - 1)];
            else if (rank < 0)
                return list[0];
            else
                return ((rank - int(rank)) * (list[unsigned(rank) + 1] - list[unsigned(rank)])) + list[unsigned(rank)];
        }


        // warning: if isSortValues is true, list will be modified
        float percentileRank(std::vector<float> &list, float value, bool sortValues)
        {
            if (sortValues)
            {
                // clean nodata
                std::vector<float> cleanList;
                for (unsigned int i = 0; i < list.size(); i++)
                {
                    if (int(list[i]) != int(NODATA))
                    {
                        cleanList.push_back(list[i]);
                    }
                }

                // check on data presence
                if (cleanList.size() < MINIMUM_PERCENTILE_DATA)
                    return NODATA;

                // sort
                quicksortAscendingFloat(cleanList, 0, unsigned(cleanList.size()-1));

                list.clear();
                list = cleanList;
            }

            float nrValuesF = float(list.size());
            unsigned int lastIndex = unsigned(list.size() - 1);

            // return rank
            if (value <= list[0]) return 0;
            if (value >= list[lastIndex]) return 100;

            for (unsigned int i = 0; i < list.size(); i++)
            {
                if (value == list[i])
                {
                    float rank = float(i + 1) / nrValuesF;
                    return rank * 100;
                }
                if (i < lastIndex && list[i] < value && list[i+1] > value)
                {
                    float rank = float(i + 1) / nrValuesF;
                    rank += (value - list[i]) / (list[i+1] - list[i]) / nrValuesF;
                    return rank * 100;
                }
            }

            return NODATA;
        }

    }

