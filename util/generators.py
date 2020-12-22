import random
from typing import List, Tuple

Point2D = Tuple[float, float]


def generate_2d_points_and_centers(
    centers_num: int = 500,
    max_distance: int = 2000,
    max_points: int = 100,
    min_points: int = 10,
    max_offset: int = 200,
) -> Tuple[List[Point2D], List[Point2D]]:

    centers = []

    for _ in range(centers_num):

        x = random.randint(-max_distance, max_distance)
        y = random.randint(-max_distance, max_distance)

        centers.append((x, y))

    points = []

    for c, (x, y) in enumerate(centers):
        for _ in range(random.randint(min_points, max_points)):
            offset_point_x = random.randint(0, max_offset)
            offset_point_y = random.randint(0, max_offset)
            points.append((x + offset_point_x, y + offset_point_y, c))

    return centers, points


def generate_2d_points(
    centers_num: int = 500,
    max_distance: int = 10000,
    max_points: int = 100,
    min_points: int = 1,
    max_offset: int = 200,
) -> List[Point2D]:

    points, _ = generate_2d_points_and_centers(centers_num,
                                               max_distance,
                                               max_points,
                                               min_points,
                                               max_offset,)
    return points
