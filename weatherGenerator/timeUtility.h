#ifndef TIMEUTILITY
#define TIMEUTILITY

    #define NRDAYSTOLERANCE  30
    #define USEDATA false

    class Crit3DDate;
    class QString;

    int getMonthsInPeriod(int month1, int month2);

    bool getDoyFromSeason(const QString &season, int myPredictionYear, int &wgDoy1, int &wgDoy2);

    bool checkLastYearDate(const Crit3DDate &inputFirstDate, const Crit3DDate &inputLastDate, int dataLength,
                           int myPredictionYear, int &wgDoy1, int &nrDaysBefore);

    void setCorrectWgDoy(int wgDoy1, int wgDoy2, int predictionYear, int myYear, int &fixedWgDoy1, int &fixedWgDoy2);


#endif // TIMEUTILITY
