from PyQt5.QtWidgets import QWidget, QGraphicsRectItem, QGraphicsEllipseItem, QApplication
from PyQt5.QtGui import QIcon, QFont, QPainter, QColor, QPen, QBrush, QPolygonF,\
QLinearGradient, QGradient
from PyQt5.QtCore import Qt, QPointF
import sys

class Screen(QWidget):
    def __init__(self, width, height, title="Untitled"):
        self.app = QApplication(sys.argv)
        super().__init__()
        self.surfaces = []
        self.fps = 30 #default is 30 fps
        self.resize(width, height)
        self.setWindowTitle(title)

    def addSurface(self, surface, x, y):
        surface.pos = (x,y)
        self.surfaces.append(surface)

    def paintEvent(self, event):
        for surface in self.surfaces:
            surface.begin(self)
            surface.render()
            surface.end()

    def keyPressEvent(self, event):
        key = event.key()
        if(key == 16777216):
            self.close()

    def quit(self):
        pass

    def start(self):
        self.show()
        sys.exit(self.app.exec_())

BLACK = (0,0,0)
WHITE = (255,255,255)

class Surface(QPainter): #TODO: change to pixmap
    def __init__(self, width, height):
        super().__init__()
        self.width, self.height = width,height
        self.x, self.y = 0,0

        self.qp = QPen()
        self.setPen(self.qp)

    @property
    def pos(self): return (self.x, self.y)

    @pos.setter
    def pos(self, pos):
        self.x, self.y = pos

    @property
    def color(self): return None

    @property
    def fillColor(self): return None

    @property
    def lineWidth(self): return None

    @color.setter
    def color(self, color):
        self.qp.setColor(QColor(*color))
        self.setPen(self.qp)

    @fillColor.setter
    def fillColor(self, color):
        self.setBrush(QBrush(QColor(*color)))

    @lineWidth.setter
    def lineWidth(self, width):
        self.qp.setWidth(width)
        self.setPen(self.qp)

    def render(self):
        self.translate(QPointF(*self.pos))
        self.filledSquare((self.x+20,self.y+20),50)
        self.filledRectangle((self.x,self.y),(50,50),WHITE)
        self.ellipse((10,10),(15,10))
        self.circle((30,30),15,(255,0,0))
        self.line((10,10),(self.x, self.y),BLACK,2)
        self.text((100,200),"ayylmao")

    def fill(self, color=WHITE):
        self.color = color
        self.filledRectangle((0,0),(self.width, self.height), color, color)

    def rectangle(self, pos, size, color=BLACK):
        self.color = color
        self.drawRect(*pos, *size)

    def filledRectangle(self, pos, size, color=BLACK, outlineColor=BLACK):
        self.fillColor = color
        self.outlineColor = outlineColor
        self.drawRect(*pos, *size)
        self.fillColor = (0,0,0,0)

    def square(self, pos, width, color=BLACK):
        self.color = color
        self.drawRect(*pos, width, width)

    def filledSquare(self, pos, width, color=BLACK, outlineColor=BLACK):
        self.fillColor = color
        self.color = outlineColor
        self.drawRect(*pos, width, width)
        self.fillColor = (0,0,0,0)

    def ellipse(self, pos, size, color=BLACK):
        self.color = color
        x,y = pos
        x -= size[0]
        y -= size[1]
        self.drawEllipse(x, y, size[0]*2, size[1]*2)

    def filledEllipse(self, pos, size, color=BLACK, outlineColor=BLACK):
        self.fillColor = color
        self.color = outlineColor
        x,y = pos
        x -= size[0]
        y -= size[1]
        self.drawEllipse(x, y, size[0]*2, size[1]*2)
        self.fillColor = (0,0,0,0)

    def circle(self, pos, rad, color=BLACK):
        self.ellipse(pos, (rad,rad), color)

    def filledCircle(self, pos, rad, color=BLACK, outlineColor=BLACK):
        self.filledEllipse(pos, (rad,rad), color, outlineColor)

    def line(self, sPos, ePos, color=BLACK, width=1):
        self.color = color
        self.lineWidth = width
        self.drawLine(*sPos,*ePos)
        self.lineWidth = 1

    def text(self, pos, text, color=BLACK):
        self.color = color
        self.drawText(*pos, text)
