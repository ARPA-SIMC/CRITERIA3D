#ifndef FORMSELECTIONSOURCE_H
#define FORMSELECTIONSOURCE_H

    #include <QtWidgets>

    class FormSelectionSource : public QDialog
    {
        Q_OBJECT

    public:
        FormSelectionSource();

        int getSourceSelectionId();
        void disableRadioButtons(bool pointDisable, bool gridDisable, bool interpolationDisable);

    private:
        QRadioButton* pointButton;
        QRadioButton* gridButton;
        QRadioButton* interpolationButton;

        void done(int res);
    };

#endif // FORMSELECTIONSOURCE_H
