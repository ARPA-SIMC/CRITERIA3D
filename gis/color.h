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
    public:
        int nrColors, nrKeyColors;
        std::vector<Crit3DColor> color, keyColor;
        float minimum, maximum;
        int classification;

        Crit3DColorScale();

        void initialize(int nrKeyColors_, int nrColors_);
        bool classify();

        Crit3DColor* getColor(float myValue);
        int getColorIndex(float myValue);
        bool setRange(float myMinimum, float myMaximum);
    };

    bool setDefaultDEMScale(Crit3DColorScale* myScale);
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


#endif // CRIT3DCOLOR_H
