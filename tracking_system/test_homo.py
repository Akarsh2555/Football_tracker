import numpy as np
import cv2
from pitch_mapper import PitchMapper

mapper = PitchMapper()
src_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
dst_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
mapper.compute_homography(src_pts, dst_pts)

points = [(1, 1), (5, 5), (9, 9)]

# 1. Old method
print("Old method:")
for p in points:
    print(mapper.transform_point(p))

# 2. Vectorized method
print("\nVectorized method:")
arr = np.array(points, dtype=np.float32)
res = mapper.transform_points(arr)
print(res)
