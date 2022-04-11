#ifndef DIALOGXMLCOMPUTATION_H
#define DIALOGXMLCOMPUTATION_H

#include <QtWidgets>

class DialogXMLComputation : public QDialog
{
    Q_OBJECT
private:
    bool isAnomaly;
    QListWidget listXMLWidget;
    QList<QString> listXML;
    int index;
public:
    DialogXMLComputation(bool isAnomaly, QList<QString> listXML);
    void elabClicked(QListWidgetItem* item);
    unsigned int getIndex() const;
};

#endif // DIALOGXMLCOMPUTATION_H
