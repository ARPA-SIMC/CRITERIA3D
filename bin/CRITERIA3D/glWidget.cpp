/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "glWidget.h"
#include "commonConstants.h"
#include "basicMath.h"

#include <cmath>

#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_0_Core>
#include <QOpenGLVertexArrayObject>


Crit3DOpenGLWidget::Crit3DOpenGLWidget(Crit3DGeometry *geometry, QWidget *parent)
    : QOpenGLWidget(parent),
    m_xRotation(0),
    m_zRotation(0),
    m_xTraslation(0),
    m_yTraslation(0),
    m_zoom(1.f),
    m_program(nullptr),
    m_geometry(geometry),
    m_vertexBuffer(QOpenGLBuffer::VertexBuffer),
    m_colorBuffer(QOpenGLBuffer::VertexBuffer)
{ }

Crit3DOpenGLWidget::~Crit3DOpenGLWidget()
{
    clear();
}


void Crit3DOpenGLWidget::clear()
{
    if (! context())
        return;

    makeCurrent();

    m_vao.destroy();

    m_vertexBuffer.destroy();
    m_colorBuffer.destroy();

    delete m_program;
    m_program = nullptr;

    doneCurrent();
}


QSize Crit3DOpenGLWidget::minimumSizeHint() const
{
    return QSize(100, 100);
}

QSize Crit3DOpenGLWidget::sizeHint() const
{
    return QSize(1000, 600);
}


static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * DEGREE_MULTIPLY;

    while (angle >= 360 * DEGREE_MULTIPLY)
        angle -= 360 * DEGREE_MULTIPLY;
}


void Crit3DOpenGLWidget::setXRotation(int angle)
{
    qNormalizeAngle(angle);

    if (angle != m_xRotation)
    {
        m_xRotation = angle;
        emit xRotationChanged(angle);
        update();
    }
}


void Crit3DOpenGLWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);

    if (angle != m_zRotation)
    {
        m_zRotation = angle;
        emit zRotationChanged(angle);
        update();
    }
}


void Crit3DOpenGLWidget::setXTraslation(float traslation)
{
    if (! isEqual(traslation, m_xTraslation))
    {
        m_xTraslation = traslation;
        update();
    }
}

void Crit3DOpenGLWidget::setYTraslation(float traslation)
{
    if (! isEqual(traslation, m_yTraslation))
    {
        m_yTraslation = traslation;
        update();
    }
}


void Crit3DOpenGLWidget::setZoom(float zoom)
{
    zoom = std::max(0.5f, zoom);

    if (! isEqual(zoom, m_zoom))
    {
        m_zoom = zoom;
        update();
    }
}


void Crit3DOpenGLWidget::setMagnify(float magnify)
{
    GLfloat newMagnify = magnify * 0.1f;

    if (isEqual(newMagnify, m_geometry->magnify()))
        return;

    m_geometry->setMagnify(newMagnify);

    if (! context())
        return;

    makeCurrent();
        m_vao.bind();
        m_vertexBuffer.bind();

        m_vertexBuffer.allocate(
            m_geometry->getVertices(),
            m_geometry->dataCount() * sizeof(GLfloat)
            );

        m_vertexBuffer.release();
        m_vao.release();
    doneCurrent();

    update();
}


static const char *vertexShaderSource =
    "#version 330 core\n"
    "layout(location = 0) in vec3 vertex;\n"
    "layout(location = 1) in vec3 color;\n"
    "out vec3 myCol;\n"
    "uniform mat4 projMatrix;\n"
    "uniform mat4 mvMatrix;\n"
    "void main() {\n"
    "   myCol = color;\n"
    "   gl_Position = projMatrix * mvMatrix * vec4(vertex,1.0);\n"
    "}\n";


static const char *fragmentShaderSource =
    "#version 330 core\n"
    "in vec3 myCol;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "   fragColor = vec4(myCol,1.0);\n"
    "}\n";


void Crit3DOpenGLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    // background: blue sky
    glClearColor(0.52f, 0.81f, 0.92f, 1.f);

    // Shader program
    m_program = new QOpenGLShaderProgram();

    bool ok = true;

    ok &= m_program->addShaderFromSourceCode(
        QOpenGLShader::Vertex,
        vertexShaderSource
        );

    ok &= m_program->addShaderFromSourceCode(
        QOpenGLShader::Fragment,
        fragmentShaderSource
        );

    ok &= m_program->link();

    if (!ok)
    {
        qDebug() << "Shader error:";
        qDebug() << m_program->log();
        return;
    }

    m_projMatrixLoc = m_program->uniformLocation("projMatrix");
    m_mvMatrixLoc = m_program->uniformLocation("mvMatrix");

    //  VAO
    m_vao.create();
    m_vao.bind();

    // Vertex buffer
    m_vertexBuffer.create();
    m_vertexBuffer.bind();

    m_vertexBuffer.allocate(
        m_geometry->getVertices(),
        m_geometry->dataCount() * sizeof(GLfloat)
        );

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        nullptr
        );

    m_vertexBuffer.release();

    // Color buffer
    m_colorBuffer.create();
    m_colorBuffer.bind();

    m_colorBuffer.allocate(
        m_geometry->getColors(),
        m_geometry->colorCount() * sizeof(GLubyte)
        );

    glEnableVertexAttribArray(1);

    glVertexAttribPointer(
        1,
        3,
        GL_UNSIGNED_BYTE,
        GL_TRUE,
        0,
        nullptr
        );

    m_colorBuffer.release();

    // VAO done
    m_vao.release();

    // Default zoom
    setZoom(m_geometry->defaultDistance());

    // OpenGL state
    glEnable(GL_DEPTH_TEST);

    // Optional:
    // glEnable(GL_CULL_FACE);

    // Optional wireframe:
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}


void Crit3DOpenGLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Camera
    m_camera.setToIdentity();
    m_camera.translate(0.f, 0.f, -m_zoom);

    // World transform
    m_world.setToIdentity();

    m_world.rotate(
        float(-m_xRotation) / DEGREE_MULTIPLY,
        1.f,
        0.f,
        0.f
        );

    m_world.rotate(
        float(m_zRotation) / DEGREE_MULTIPLY,
        0.f,
        0.f,
        1.f
        );

    m_world.translate(
        m_xTraslation,
        m_yTraslation,
        0.f
        );

    // Render
    m_program->bind();

    m_program->setUniformValue(
        m_projMatrixLoc,
        m_proj
        );

    m_program->setUniformValue(
        m_mvMatrixLoc,
        m_camera * m_world
        );

    m_vao.bind();

    glDrawArrays(
        GL_TRIANGLES,
        0,
        m_geometry->vertexCount()
        );

    m_vao.release();

    m_program->release();
}


void Crit3DOpenGLWidget::resizeGL(int w, int h)
{
    m_proj.setToIdentity();

    GLfloat aspect = (h == 0) ? 1.0 : GLfloat(w) / GLfloat(h);
    GLfloat farPlane = std::max(10000.f, m_geometry->defaultDistance() * 2.f);

    m_proj.perspective(45.f, aspect, 1.f, farPlane);
}

void Crit3DOpenGLWidget::mousePressEvent(QMouseEvent *event)
{
    m_lastPos = event->pos();
}


void Crit3DOpenGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->pos().x() - m_lastPos.x();
    int dy = event->pos().y() - m_lastPos.y();

    if (event->buttons() & Qt::LeftButton)
    {
        float angle = float(m_zRotation) / DEGREE_MULTIPLY * DEG_TO_RAD;

        float cosAngle = std::cos(angle);
        float sinAngle = std::sin(angle);

        setXTraslation(m_xTraslation + m_zoom * (dx * cosAngle - dy * sinAngle) * 0.002f);
        setYTraslation(m_yTraslation - m_zoom * (dy * cosAngle + dx * sinAngle) * 0.002f);
    }
    else if (event->buttons() & Qt::RightButton)
    {
        setXRotation(m_xRotation - dy * DEGREE_MULTIPLY * 0.5f);
        setZRotation( m_zRotation + dx * DEGREE_MULTIPLY * 0.5f);
    }

    m_lastPos = event->pos();
}


void Crit3DOpenGLWidget::wheelEvent(QWheelEvent *event)
{
    int dy = event->angleDelta().y();

    setZoom(m_zoom * std::pow(0.995f, dy / 8.f));
    //setZoom(m_zoom * std::pow(0.999f, dy));
}

