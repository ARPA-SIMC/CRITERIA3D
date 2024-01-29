#ifndef CRIT3DCOLOR_H
#define CRIT3DCOLOR_H

    #include <vector>

    namespace classificationMethod
    {
        enum type{EqualInterval, Gaussian, Quantile, Categories, UserDefinition };
    }

    class Crit3DColor {
    public:
        short red;
        short green;
        short blue;

        Crit3DColor();
        Crit3DColor(short,short,short);
    };

    class Crit3DColorScale {

    private:
        unsigned int _nrColors, _nrKeyColors;
        std::vector<Crit3DColor> color;
        float _minimum, _maximum;
        bool _isRangeBlocked;
        int _classification;

    public:
        std::vector<Crit3DColor> keyColor;

        Crit3DColorScale();

        void initialize(unsigned int nrKeyColors, unsigned int nrColors);
        bool classify();

        unsigned int nrColors() { return _nrColors; }
        unsigned int nrKeyColors() { return _nrKeyColors; }

        float minimum() { return _minimum; }
        void setMinimum(float min) { _minimum = min; }

        float maximum() { return _maximum; }
        void setMaximum(float max) { _maximum = max; }

        Crit3DColor* getColor(float myValue);
        unsigned int getColorIndex(float myValue);

        bool setRange(float minimum, float maximum);
        void setRangeBlocked(bool blocked) { _isRangeBlocked = blocked; }
        bool isRangeBlocked() { return _isRangeBlocked; }
    };

    bool setDefaultScale(Crit3DColorScale* myScale);
    bool setDTMScale(Crit3DColorScale* myScale);
    bool setTemperatureScale(Crit3DColorScale* myScale);
    bool setAnomalyScale(Crit3DColorScale* myScale);
    bool setPrecipitationScale(Crit3DColorScale* myScale);
    bool setRelativeHumidityScale(Crit3DColorScale* myScale);
    bool setRadiationScale(Crit3DColorScale* myScale);
    bool setWindIntensityScale(Crit3DColorScale* myScale);
    bool setCenteredScale(Crit3DColorScale* myScale);
    bool setCircolarScale(Crit3DColorScale* myScale);
    bool roundColorScale(Crit3DColorScale* myScale, int nrIntervals, bool lessRounded);
    bool reverseColorScale(Crit3DColorScale* myScale);
    bool setGrayScale(Crit3DColorScale* myScale);
    bool setBlackScale(Crit3DColorScale* myScale);
    bool setSurfaceWaterScale(Crit3DColorScale* myScale);
    bool setLAIScale(Crit3DColorScale* myScale);


#endif // CRIT3DCOLOR_H
