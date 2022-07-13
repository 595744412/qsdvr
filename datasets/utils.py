class Camera(object):
    def __init__(
        self, width, height, rotation, translation, scale, fx, fy, cx, cy, near, far
    ):
        self.width = width
        self.height = height
        self.rotation = rotation
        self.translation = translation
        self.scale = scale
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near = near
        self.far = far

    def generate_ray_points():
        return
