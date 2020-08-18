#include "tableDelegate.h"
#include <QLineEdit>
#include <QDoubleValidator>

TableDelegate::TableDelegate(QObject *parent) : QStyledItemDelegate(parent)
{
}

QWidget* TableDelegate::createEditor(QWidget* parent,const QStyleOptionViewItem &option,const QModelIndex &index) const
{
    Q_UNUSED(option);
    Q_UNUSED(index);

    QLineEdit* editor = new QLineEdit(parent);
    double bottom = 0;
    double top = 1000000;
    int d = 5;
    myValidator* val = new myValidator(bottom, top, d, editor);
    editor->setValidator(val);
    return editor;
}

void TableDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    double value = index.model()->data(index,Qt::EditRole).toDouble();
    QLineEdit* line = static_cast<QLineEdit*>(editor);
    line->setText(QString().setNum(value));

}

void TableDelegate::setModelData(QWidget* editor,QAbstractItemModel* model,const QModelIndex &index) const
{
    QLineEdit* line = static_cast<QLineEdit*>(editor);
    QString value = line->text();

    if (value.isEmpty())
    {
        model->setData(index, EMPTYVALUE);
    }
    else
    {
        model->setData(index,value);
    } 

}

void TableDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(index);
    editor->setGeometry(option.rect);
}

// myValidator Constructor
myValidator::myValidator ( double bottom, double top, int decimals,
        QObject* parent = 0 )
    : QDoubleValidator ( bottom, top, decimals, parent ) {

}

// Custom validate function to allow empty values
QValidator::State myValidator::validate ( QString& input, int& ) const {

    const double b = bottom();
    const double t = top();
    const int d = decimals();

    // Check if the input is empty
    if ( input.isEmpty() ) {
        return Acceptable;
    }
    else
    {
        bool ok;
        double entered = input.toDouble ( &ok );
        if ( !ok && !input.size() ) {
            // Handle an edge case where there are no characters left.
            ok = true;
        } else if ( !ok && ( input == "." ) ) {
            // If the only character is a decimal, then it is fine.
            ok = true;
        }

        if ( !ok || entered > t || entered < b ) {
            // Not a number or out of range, so invalid
            return Invalid;
        } else if ( input.contains ( "." ) ) { // If it cant, check for a decimal point and how many numbers follow
            QStringList dec = input.split ( "." );
            if ( dec[1].size() > d ) {
                // Too many decimals
                return Invalid;
            }
        }
    }
    return Acceptable;
}
