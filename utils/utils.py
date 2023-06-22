import numpy as np

def make_constraint(polygon):
    """
    Function to create a new function for checking points within a specific polygon.

    Args:
    polygon (list): List of tuples representing the vertices of the polygon in counter-clockwise order

    Returns:
    function: Function that takes a point (x, y) and returns True if the point is in the polygon, False otherwise
    """

    num_vertices = len(polygon)

    def point_in_polygon(state):
        x, y, _, _ = state
        """
        Function to determine if a point is inside the specific polygon using the Ray Casting algorithm.

        Args:
        x, y (float): Coordinates of the point to test

        Returns:
        bool: True if the point is in the polygon, False otherwise
        """

        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, num_vertices + 1):
            p2x, p2y = polygon[i % num_vertices]
            if min(p1y, p2y) < y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersects = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_intersects:
                        inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    return point_in_polygon

def line_search(m, h, c):
    start = m
    end = h

    while np.linalg.norm(start - end) > 1e-6:
        mid = (start + end) / 2
        if c(mid):
            start = mid
        else:
            end = mid

    return end

def nearest_point(x0, y0, c, precision=0.01, max_dist=1000):
    radius = 0

    while radius < max_dist:
        for theta in np.linspace(0, 2 * math.pi, 100):
            x = x0 + radius * np.cos(theta)
            y = y0 + radius * np.sin(theta)
            if c([x, y, 0, 0]):
                # Get the unit vector from (x, y) to (x0, y0)
                unit_vector = np.array([x0 - x, y0 - y]) / np.linalg.norm(np.array([x0 - x, y0 - y]))
                x_tmp = x + unit_vector[0] * precision
                y_tmp = y + unit_vector[1] * precision                
                
                return line_search(np.array([x_tmp, y_tmp, 0, 0]), np.array([x, y, 0, 0]), c)[:2]
                
        radius += precision