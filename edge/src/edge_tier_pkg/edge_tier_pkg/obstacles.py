import math

class Obstacle:
    """
    This class defines the positions of obstacles to collision avoidance purposes.
    If the obstacle has a box shape, for example, this class will divide the box into cylinders.
    The radius of these cylinders will be take an account by MPC to avoid collisions.
    The obstacles are generated acoording to the loaded scenario and matchs the obstacles in Gazebo.
    """

    obstacles_x = []
    obstacles_y = []
    obstacles_diameter = []

    def __init__(self, **kwargs):
        x = kwargs.get('x')
        y = kwargs.get('y')
        theta = kwargs.get('theta')
        self.len = kwargs.get('length')
        self.diam = kwargs.get('diameter')
        self.shape = kwargs.get('shape')

        if self.shape == 'box':
            self.pose = [x, y, math.radians(theta)]
            self.size = (self.len/self.diam) / 2
            self.x = []
            self.y = []

            x_ref = self.pose[0] - self.diam
            y_ref = self.pose[1] - self.diam
            for i in range(int(self.size)):
                if self.pose[2] == math.radians(90.0):
                    self.x.append(self.pose[0])
                else:
                    self.x.append(x_ref*math.cos(self.pose[2]))
                if self.pose[2] == 0.0:
                    self.y.append(self.pose[1])
                else:
                    self.y.append(y_ref*math.sin(self.pose[2]))
                x_ref = x_ref - self.diam
                y_ref = y_ref - self.diam

            self.x.reverse()
            self.y.reverse()

            self.x.append(self.pose[0])
            self.y.append(self.pose[1])

            x_ref = self.pose[0] + self.diam
            y_ref = self.pose[1] + self.diam
            for i in range(int(self.size)):
                if self.pose[2] == math.radians(90.0):
                    self.x.append(self.pose[0])
                else:
                    self.x.append(x_ref * math.cos(self.pose[2]))
                if self.pose[2] == 0.0:
                    self.y.append(self.pose[1])
                else:
                    self.y.append(y_ref * math.sin(self.pose[2]))
                x_ref = x_ref + self.diam
                y_ref = y_ref + self.diam

            Obstacle.save_obstacles(self.shape, self.x, self.y, self.diam)

        if self.shape == 'cylinder':
            Obstacle.save_obstacles(self.shape, x, y, self.diam)

    @classmethod
    def save_obstacles(cls, shape, x, y, diameter):
        if shape == 'box':
            for item in x:
                cls.obstacles_x.append(item)
                cls.obstacles_diameter.append(diameter)
            for item in y:
                cls.obstacles_y.append(item)
        if shape == 'cylinder':
            cls.obstacles_x.append(x)
            cls.obstacles_y.append(y)
            cls.obstacles_diameter.append(diameter)


def load_obstacles(scenario):

    if scenario == 1: # cenario demonstrativo
        cylinder_1 = Obstacle(shape='cylinder', x=0.0, y=0.0, diameter=1.0)
        cylinder_2 = Obstacle(shape='cylinder', x=5.0, y=5.0, diameter=1.0)
        cylinder_3 = Obstacle(shape='cylinder', x=-5.0, y=5.0, diameter=1.0)
        cylinder_3 = Obstacle(shape='cylinder', x=-5.0, y=-5.0, diameter=1.0)
        cylinder_4 = Obstacle(shape='cylinder', x=5.0, y=-5.0, diameter=1.0)

    if scenario == 2: # pequena linha de procucao
        ws_1 = Obstacle(shape='box', x=4.0, y=2.0, theta=0.0, diameter=1.0, length=3.0)
        ws_2 = Obstacle(shape='box', x=4.0, y=-2.0, theta=0.0, diameter=1.0, length=3.0)
        ws_3 = Obstacle(shape='box', x=-4.0, y=2.0, theta=0.0, diameter=1.0, length=3.0)
        ws_4 = Obstacle(shape='box', x=-4.0, y=-2.0, theta=0.0, diameter=1.0, length=3.0)
        warehouse = Obstacle(shape='box', x=25.0, y=0.0, theta=0.0, diameter=5.0*math.sqrt(2), length=5.0)
        fp = Obstacle(shape='box', x=-19.0 , y=0.0, theta=0.0, diameter=5.0*math.sqrt(2), length=5.0)

    if scenario == 3: # logistica
        d = 1.25
        d2 = d

        # Hangar 1
        x = -37.5
        y = -30.0
        hangar_1_1 = Obstacle(shape='box', x=x+0.0, y=y+12.5, theta=0.0, diameter=d2, length=25.0)
        hangar_1_2_1 = Obstacle(shape='box', x=x+12.5, y=y+12.5-3.75, theta=90.0, diameter=d, length=7.5)
        hangar_1_2_2 = Obstacle(shape='box', x=x+12.5, y=y-12.5+3.75, theta=90.0, diameter=d, length=7.5)
        hangar_1_3 = Obstacle(shape='box', x=x-12.5, y=y, theta=90.0, diameter=d, length=25.0)
        hangar_1_4 = Obstacle(shape='box', x=x+0.0, y=y-12.4249, theta=0.0, diameter=d2, length=25.0)

        # Hangar 2
        x = -37.5
        y = 0.0
        hangar_2_1 = Obstacle(shape='box', x=x+0.0, y=y+12.5, theta=0.0, diameter=d2, length=25.0)
        hangar_2_2_1 = Obstacle(shape='box', x=x+12.5, y=y+12.5-3.75, theta=90.0, diameter=d, length=7.5)
        hangar_2_2_2 = Obstacle(shape='box', x=x+12.5, y=y-12.5+3.75, theta=90.0, diameter=d, length=7.5)
        hangar_2_3 = Obstacle(shape='box', x=x-12.5, y=y, theta=90.0, diameter=d, length=25.0)
        hangar_2_4 = Obstacle(shape='box', x=x+0.0, y=y-12.4249, theta=0.0, diameter=d2, length=25.0)

        # Hangar 3
        x = -37.5
        y = 30.0
        hangar_3_1 = Obstacle(shape='box', x=x+0.0, y=y+12.5, theta=0.0, diameter=d2, length=25.0)
        hangar_3_2_1 = Obstacle(shape='box', x=x+12.5, y=y+12.5-3.75, theta=90.0, diameter=d, length=7.5)
        hangar_3_2_2 = Obstacle(shape='box', x=x+12.5, y=y-12.5+3.75, theta=90.0, diameter=d, length=7.5)
        hangar_3_3 = Obstacle(shape='box', x=x-12.5, y=y, theta=90.0, diameter=d, length=25.0)
        hangar_3_4 = Obstacle(shape='box', x=x+0.0, y=y-12.4249, theta=0.0, diameter=d2, length=25.0)

        # Dock 1
        dock_1 = Obstacle(shape='box', x=0.0, y=-95.0, theta=0.0, diameter=5.0, length=25.0)

        # Dock 2
        dock_2 = Obstacle(shape='box', x=45.0, y=-50.0, theta=90.0, diameter=5.0, length=25.0)

        # Dock 3
        dock_3 = Obstacle(shape='box', x=45.0, y=50.0, theta=90.0, diameter=5.0, length=25.0)

        # Dock 4
        dock_4 = Obstacle(shape='box', x=0.0, y=95.0, theta=0.0, diameter=5.0, length=25.0)

    if scenario == 4:
        pass