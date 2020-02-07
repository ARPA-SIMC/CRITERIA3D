#ifndef TIMEUTILITY
#define TIMEUTILITY

    #define NRDAYSTOLERANCE  30
    #define USEDATA false

    class Crit3DDate;
    class QString;

    int numMonthsInPeriod(int m1, int m2);

    bool getDoyFromSeason(QString season, int myPredictionYear, int* wgDoy1, int* wgDoy2);

    bool checkLastYearDate(Crit3DDate inputFirstDate, Crit3DDate inputLastDate, int dataLenght, int myPredictionYear, int* wgDoy1, int *nrDaysBefore);

    void fixWgDoy(int wgDoy1, int wgDoy2, int predictionYear, int myYear, int* fixwgDoy1, int* fixwgDoy2);


#endif // TIMEUTILITY
