#ifndef DIALOGPRAGASETTINGS_H
#define DIALOGPRAGASETTINGS_H

#include <QWidget>

#ifndef PRAGAPROJECT_H
    #include "pragaProject.h"
#endif

#ifndef ELABORATIONSETTINGS_H
    #include "elaborationSettings.h"
#endif

#ifndef DIALOGSETTINGS_H
    #include "dialogSettings.h"
#endif

class ElaborationTab : public QWidget
{
    Q_OBJECT

public:
    explicit ElaborationTab(Crit3DElaborationSettings *elabSettings);

    QLineEdit anomalyPtsMaxDisEdit;
    QLineEdit anomalyPtsMaxDeltaZEdit;
    QLineEdit gridMinCoverageEdit;
    QCheckBox mergeJointStationsEdit;

private:
};

class DialogPragaSettings : public DialogSettings
{
    Q_OBJECT

    public:
        explicit DialogPragaSettings(PragaProject* myProject);

        bool acceptPragaValues();
        void accept();

    protected:
        PragaProject* project_;

    private:
        Crit3DElaborationSettings *_elabSettings;
        ElaborationTab* elabTab;
};


#endif // DIALOGPRAGASETTINGS_H
