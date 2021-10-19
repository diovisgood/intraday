import sys
from typing import Optional, Tuple, Sized, Literal, Sequence
from numbers import Real
import numpy as np
import pyglet
from pyglet.gl import *
from gym.envs.classic_control.rendering import Geom, LineWidth

RAD2DEG = 57.29577951308232


def get_display(spec):
    """
    Convert a display specification (such as :0) into an actual Display object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise RuntimeError(f'Invalid display specification: {spec}. (Must be a string like :0 or None.)')


def get_window(width, height, display, **kwargs):
    """Will create a pyglet window from the display specification provided."""
    screen = display.get_screens() #available screens
    config = screen[0].get_best_config() #selecting the first screen
    context = config.create_context(None) #create GL context
    return pyglet.window.Window(width=width, height=height, display=display, config=config, context=context, **kwargs)


class Viewer(object):
    def __init__(self, width, height, display=None, resizable=False):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = get_window(width=width, height=height, display=display, resizable=resizable)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def window_closed_by_user(self):
        self.isopen = False

    def add_geom(self, geom):
        self.geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        for geom in self.geoms:
            geom.render()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1, :, 0:3]

    def __del__(self):
        self.close()


class Geom(object):
    def __init__(self, color: Optional[Tuple] = None):
        self.attrs = []
        if color is not None:
            self.attrs.append(Color(color))

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
            
    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Translate(Attr):
    def __init__(self, dx, dy):
        self.translation = (float(dx), float(dy))

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0)

    def disable(self):
        glPopMatrix()

    def set_translation(self, dx, dy):
        self.translation = (float(dx), float(dy))
        

class Scale(Attr):
    def __init__(self, sx, sy):
        self.scale = (float(sx), float(sy))

    def enable(self):
        glPushMatrix()
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()

    def set_scale(self, sx, sy):
        self.scale = (float(sx), float(sy))


class Rotate(Attr):
    def __init__(self, degree=0.0):
        self.degree = float(degree)

    def enable(self):
        glPushMatrix()
        glRotatef(RAD2DEG * self.degree, 0, 0, 1.0)
        
    def disable(self):
        glPopMatrix()

    def set_rotation(self, degree):
        self.degree = float(degree)
        
        
class Color(Attr):
    def __init__(self, color: Tuple):
        self.color = _color_f4(color)

    def enable(self):
        glColor4f(*self.color)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self):
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, line_width):
        self.line_width = line_width

    def enable(self):
        glLineWidth(self.line_width)


class FilledPolygon(Geom, Sized):
    def __init__(self, v: Sequence[Tuple], color: Optional[Tuple] = None):
        Geom.__init__(self, color=color)
        self.v = v

    def __len__(self):
        return len(self.v)

    def render1(self):
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()


class Group(Geom, Sized):
    def __init__(self):
        super().__init__()
        self.geoms = []
    
    def __len__(self):
        return len(self.geoms)

    def reset(self):
        self.geoms.clear()

    def add_geom(self, geom):
        self.geoms.append(geom)

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        for geom in self.geoms:
            geom.render()
        for attr in self.attrs:
            attr.disable()


class Line(Geom):
    def __init__(self,
                 p1=(0.0, 0.0),
                 p2=(0.0, 0.0),
                 color: Optional[Tuple] = None,
                 line_width: Optional[int] = None):
        Geom.__init__(self, color=color)
        self.p1 = p1
        self.p2 = p2
        if isinstance(line_width, Real):
            self.add_attr(LineWidth(line_width))

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.p1)
        glVertex2f(*self.p2)
        glEnd()
        

class LineChart(Geom):
    def __init__(self,
                 point: Optional[Tuple[Real, Real]] = None,
                 color: Optional[Tuple] = None,
                 line_width: Optional[int] = None):
        Geom.__init__(self, color=color)
        self.v = []
        self.last_color: Optional[Tuple[float, float, float, float]] = None
        if point is not None:
            self.line_to(point, color)
        elif color is not None:
            color = _color_f4(color)
            self.v.append(color)
        if isinstance(line_width, Real):
            self.add_attr(LineWidth(line_width))
    
    def reset(self):
        self.v.clear()
        self.last_color = None
        
    def line_to(self, point: Tuple[Real, Real], color: Optional[Tuple] = None):
        if color is not None:
            color = _color_f4(color)
            if (self.last_color is None) or (self.last_color != color):
                self.v.append(color)
                self.last_color = color
        assert isinstance(point, Tuple) and (len(point) == 2)
        self.v.append(point)
        # self.v.append(self.last_color)

    def render1(self):
        glBegin(GL_LINE_STRIP)
        for p in self.v:
            if len(p) == 4:
                glColor4f(*p)
            else:
                glVertex3f(p[0], p[1], 0)
        glEnd()
    

class Label(Geom):
    def __init__(self,
                 text: str,
                 pos: Tuple[Real, Real],
                 font_name: str = 'Arial',
                 font_size: Real = 10,
                 bold=False,
                 italic=False,
                 color: Tuple = (255, 0, 0, 255),
                 anchor_x: Literal['left', 'center', 'right'] = 'left',
                 anchor_y: Literal['top', 'center', 'baseline', 'bottom'] = 'baseline',
                 align: Literal['left', 'right', 'center'] = 'left'
                 ):
        super().__init__()
        self.pos = pos
        self.label = pyglet.text.Label(
            text=text,
            font_name=font_name,
            font_size=font_size,
            bold=bold,
            italic=italic,
            color=_color_b4(color),
            x=pos[0],
            y=pos[1],
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            align=align,
        )
    
    @property
    def text(self):
        return self.label.text
    
    @text.setter
    def text(self, text: str):
        self.label.text = text
    
    @property
    def color(self):
        return self.label.color
    
    @color.setter
    def color(self, c: Tuple):
        self.label.color = _color_b4(c)
    
    def render1(self):
        self.label.draw()
        

def _color_b4(c: Tuple) -> Tuple[int, int, int, int]:
    assert isinstance(c, Tuple) and (len(c) in {3, 4})
    if all([(isinstance(x, float) and (0.0 <= x <= 1.0)) for x in c]):
        c = tuple(int(max(0.0, min(255.0, 255.0 * x))) for x in c)
    if len(c) < 4:
        c = c + (255,)
    return c


def _color_f4(c: Tuple) -> Tuple[int, int, int, int]:
    assert isinstance(c, Tuple) and (len(c) in {3, 4})
    if all([(isinstance(x, int) and (0 <= x <= 255)) for x in c]):
        c = tuple(max(0.0, min(1.0, x / 255.0)) for x in c)
    if len(c) < 4:
        c = c + (1.0,)
    return c
