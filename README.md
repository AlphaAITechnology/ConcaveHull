# ConcaveHull
This is a function based on convex hull implementation, while limiting the maximum distance for searching next available node.

The function currently is implementated for assisting the computer vision annotation, hence the current version only works on input points range in 5000*5000.

## version 1.0
Initial commit that could handle shape of 5000 * 5000 using cv2 to optimize the performance.

### Main function:
### ConcaveHull (points, distance = 10)
Input

Points: (required)

type - numpy array with dtype numpy int64

description - Points to be used for the concave hull calculation

e.g. np.array([[0,0],[0,1],[3,2],...])

Distance: (optional, default 10)

type - int (or list of ints, tobe implemented)

description - Distance to be considered for the neighbourhood, such that the concave shape could be formed while constructing the hull.

Return

Points:

type - numpy array with dtype numpy int64

decription - List of points represent the points in the concave hull. The return list is arranged in the order of points constructing the hull.
