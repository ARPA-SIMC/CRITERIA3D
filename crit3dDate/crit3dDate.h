#ifndef CRIT3DDATE_H
#define CRIT3DDATE_H

    #include <string>

    #ifndef HOUR_SECONDS
        #define HOUR_SECONDS 3600
    #endif

    #ifndef DAY_SECONDS
        #define DAY_SECONDS 86400
    #endif

    #ifndef NO_DATE
        #define NO_DATE Crit3DDate(0,0,0)
    #endif

    #ifndef NO_DATETIME
        #define NO_DATETIME Crit3DTime(NO_DATE, 0)
    #endif

    class Crit3DDate
    {
    public:
        int day;
        int month;
        int year;

        Crit3DDate();
        Crit3DDate(int myDay, int myMonth, int myYear);
        Crit3DDate(std::string myDate);

        friend bool operator == (const Crit3DDate& myFirstDate, const Crit3DDate& mySecondDate);
        friend bool operator != (const Crit3DDate& myFirstDate, const Crit3DDate& mySecondDate);
        friend bool operator >  (const Crit3DDate& myFirstDate, const Crit3DDate& mySecondDate);
        friend bool operator >= (const Crit3DDate& myFirstDate, const Crit3DDate& mySecondDate);
        friend bool operator <  (const Crit3DDate& myFirstDate, const Crit3DDate& mySecondDate);
        friend bool operator <= (const Crit3DDate& myFirstDate, const Crit3DDate& mySecondDate);

        friend Crit3DDate& operator ++ (Crit3DDate& myFirstDate);
        friend Crit3DDate& operator -- (Crit3DDate& myFirstDate);

        void setDate(int myDay, int myMonth, int myYear);
        bool isNullDate();
        void setNullDate();

        Crit3DDate addDays(long offset) const;
        int daysTo(const Crit3DDate& newDate) const;

        std::string toISOString() const;

        std::string toString() const;
    };


    class Crit3DTime
    {
    public:
        Crit3DDate date;
        int time;               // [s]

        Crit3DTime();
        Crit3DTime(const Crit3DDate &myDate, int myTime);

        friend bool operator > (const Crit3DTime& myFirstTime, const Crit3DTime& mySecondTime);
        friend bool operator < (const Crit3DTime& myFirstTime, const Crit3DTime& mySecondTime);
        friend bool operator >= (const Crit3DTime& myFirstTime, const Crit3DTime& mySecondTime);
        friend bool operator <= (const Crit3DTime& myFirstTime, const Crit3DTime& mySecondTime);
        friend bool operator == (const Crit3DTime& myFirstTime, const Crit3DTime& mySecondTime);
        friend bool operator != (const Crit3DTime& myFirstTime, const Crit3DTime& mySecondTime);

        Crit3DTime addSeconds(long mySeconds) const;
        bool isEqual(const Crit3DTime&) const;
        bool isNullTime();

        void setNullTime();

        bool setFromISOString(const std::string &dateTimeStr);

        int getHour() const;
        int getNearestHour() const;
        int getMinutes() const;
        int getSeconds() const;

        int hourTo(const Crit3DTime &newTime);

        std::string toISOString() const;
        std::string toString() const;
    };

    bool isLeapYear(int year);
    int getDaysInMonth(int month, int year);

    int getDoyFromDate(const Crit3DDate& myDate);
    int getMonthFromDoy(int doy,int year);

    Crit3DDate getDateFromDoy(int year, int doy);
    Crit3DDate getDateFromDoyGeneric(int year, int doy);

    Crit3DDate max(const Crit3DDate& myDate1, const Crit3DDate& myDate2);
    Crit3DDate min(const Crit3DDate& myDate1, const Crit3DDate& myDate2);

    int difference(const Crit3DDate &firstDate, const Crit3DDate &lastDate);

    inline long getJulianDay(int day, int month, int year);
    Crit3DDate getDateFromJulianDay(long julianDay);

    int daysTo(const Crit3DDate& firstDate, const Crit3DDate& lastDate);


#endif // CRIT3DDATE_H
