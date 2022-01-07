##################################################################
# Concave Hull
# obtain the concave hull of points that join points within
# a certain distance, default distance of 10 in radius
# it is an O(n^2) algorithm
# This method used cv2 to optimize the runtime,
# currently the input limitation is size of 25,000,000 (5000*5000)
##################################################################
# Main function:
# ConcaveHull (points, distance = 10)
# - Input
# Points: (required)
# type - numpy array with dtype numpy int64
# description - Points to be used for the concave hull calculation
# e.g. np.array([[0,0],[0,1],[3,2],...])
#
# Distance: (optional, default 10)
# type - int (or list of ints, tobe implemented)
# description - Distance to be considered for the neighbourhood,
#               such that the concave shape could be formed while
#               constructing the hull.
# - Return
# Points:
# type - numpy array with dtype numpy int64
# decription - List of points represent the points in the concave
#              hull. The return list is arranged in the order of
#              points constructing the hull.
##################################################################

import cv2, math
import numpy as np
from operator import add

# Adding elements in list a and list b
def liAdd(a,b):
    return list(map(add,a,b))

# Get the index of point within a list 
def nextNodeIndex(neighList,prevPtAngle):
    return np.searchsorted(neighList, prevPtAngle, side='right')

# detect if any head of AB, CD lies on the other line
def collinearPoints(ax,ay,bx,by,cx,cy,dx,dy):
    return (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by) == 0 and (ax != cx or ay != cy) and (bx != cx or by != cy) and (ax == cx or (ax - bx)/(ax - cx) > 1) and (ay == cy or (ay - by)/(ay - cy) > 1)) or \
            (ax * (by - dy) + bx * (dy - ay) + dx * (ay - by) == 0 and (ax != dx or ay != dy) and (bx != dx or by != dy) and (ax == dx or (ax - bx)/(ax - dx) > 1) and (ay == dy or (ay - by)/(ay - dy) > 1)) or \
            (ax * (cy - dy) + cx * (dy - ay) + dx * (ay - cy) == 0 and (ax != cx or ay != cy) and (ax != dx or ay != dy) and (cx == ax or (cx - dx)/(cx - ax) > 1) and (cy == ay or (cy - dy)/(cy - ay) > 1)) or \
            (bx * (cy - dy) + cx * (dy - by) + dx * (by - cy) == 0 and (bx != cx or by != cy) and (bx != dx or by != dy) and (cx == bx or (cx - dx)/(cx - bx) > 1) and (cy == by or (cy - dy)/(cy - by) > 1))

def ccw(ax,ay,bx,by,cx,cy):
    return (cy-ay) * (bx-ax) > (by-ay) * (cx-ax)

# Return true if lines intersect or concurrent in same direction
def intersect(ax,ay,bx,by,cx,cy,dx,dy):
    return (ccw(ax,ay,cx,cy,dx,dy) != ccw(bx,by,cx,cy,dx,dy) and ccw(ax,ay,bx,by,cx,cy) != ccw(ax,ay,bx,by,dx,dy) and ccw(ax,ay,dx,dy,cx,cy) != ccw(bx,by,dx,dy,cx,cy) and ccw(ax,ay,bx,by,dx,dy) != ccw(ax,ay,bx,by,cx,cy)) or collinearPoints(ax,ay,bx,by,cx,cy,dx,dy)

def ConcaveHull(points, distList = [10]):
    assert isinstance(points,np.ndarray), "Type error. Input points should be numpy.ndarray format."
    assert len(points.shape) == 2 and points.shape[1] == 2, "Type error. ndarray not in expected shape. (_, 2)"
    assert points.dtype == np.int64, "Type error. This function currently handle int64 dtype only."
    
    if isinstance(distList,(int,float)):
        distList = [distList]
    elif isinstance(distList,np.ndarray):
        distList = distList.tolist()
    assert isinstance(distList,list), "Type error. Distance list should be either a list or a number."
    
    # normalizing the input points
    xs, ys = points[:,0], points[:,1]
    minX, minY, row, col = min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    normXs, normYs = xs - minX, ys - minY
    
    # Temporary we will handle 5000 * 5000 pixels size with graphic approach
    assert row * col <= 5000 * 5000, "Unsupported. Currently we only handle 5000*5000 pixels operation."
    
    emArr = np.zeros((row,col),np.float32)
    ptArr = np.zeros((row,col),bool)
    ptArr[normXs,normYs] = True
    
    ptNeigh = {}
    topLeft = 0
    TLx,TLy = normXs[0],normYs[0]
    
    for i in range(len(normXs)):
        if normXs[i] < TLx:
            topLeft = i
            TLx = normXs[i]
            TLy = normYs[i]
        if normXs[i] == TLx:
            if normYs[i] < TLy:
                topLeft = i
                TLy = normYs[i]
        
        # image axis is tranposed
        imPt = (normYs[i],normXs[i])
        
        neigh = []
        captured = np.zeros((row,col),bool)
        timeDiff = 0
        for d in distList:
            # cv2 does not accept bool array, so we use float32
            current = np.zeros((row,col),np.float32)
            cv2.circle(current, imPt, d,1,-1)
            
            current = current.astype(bool) ^ captured
            captured = captured | current
            current[normXs[i]][normYs[i]] = False
            
            xys = list(np.where(current & ptArr))
            
            # add angle between points
            xys.append(np.array([(math.atan2(xys[0][i] - imPt[1], xys[1][i] - imPt[0])) % (math.pi * 2) 
                                for i in range(len(xys[0])) ]))
            
            # add distance between points
            xys.append(np.array([math.sqrt((xys[0][i] - imPt[1]) ** 2 + (xys[1][i] - imPt[0]) ** 2)
                                for i in range(len(xys[0])) ]))
            pts = np.dstack(xys)[0]
            
            # sort with ascending angle then ascending distance
            sortPts = pts[np.lexsort((pts[:,3],pts[:,2]))]
            neigh.append(sortPts)
        ptNeigh[tuple(np.flip(imPt))] = neigh
    
    # Reduce the runtime of seeking potential collision lines
    setArrAdd = lambda x, y: [*map(lambda z: z.add(y), x)]
    collisionSet = np.array([set() for _ in range(row * col)], dtype = object).reshape((row,col))
    
    iniPt = prevPt = (normXs[topLeft],normYs[topLeft])
    line = []
    prevAngle = np.pi
    curPt = ''
    
    # limit the maximum iteration, considering the worse case to be visiting each node 2 times, and some particular node visited more than 2 times, but less than the number of nodes.
    brk = len(points) * 3
    
    while brk > 0:
        brk -= 1
        if tuple(curPt) == iniPt:
            break
        for i in range(len(distList)):
            if len(ptNeigh[prevPt][i]) < 1: # skip empty neighbour
                continue
            nextIndex = nextNodeIndex(np.round(ptNeigh[prevPt][i][...,2],6),np.round(prevAngle,6)) % len(ptNeigh[prevPt][i])
            for j in range(len(ptNeigh[prevPt][i])):
                curPt = ptNeigh[prevPt][i][(nextIndex + j) % len(ptNeigh[prevPt][i])][:2]
                curPt = tuple(curPt.astype(int).tolist())
                curAng = ptNeigh[prevPt][i][(nextIndex + j) % len(ptNeigh[prevPt][i])][2]
                
                lineArea = np.zeros((row,col))
                cv2.line(lineArea, prevPt, curPt, 1, 2)
                pts = np.where(lineArea)
                lineIndex = list(set().union(*collisionSet[pts]))
                
                valid = 1 # 1: Valid, 0: Invalid, -1: Duplicate
                for li in np.array(line)[lineIndex]:
                    if collinearPoints(*prevPt,*curPt,*li) and prevPt == list(li[:2]):
                        # duplicates
                        valid = -1
                        break
                    if intersect(*prevPt,*curPt,*li):
                        # crossover
                        valid = 0
                        break
                if valid > 0:
                    break
            if valid > 0:
                # draw and add aware area for next collision detection
                lineArea = np.zeros((row,col))
                cv2.line(lineArea, prevPt, curPt, 1, 2)
                pts = np.where(lineArea)
                setArrAdd(collisionSet[pts],len(line))
                
                line.append([*prevPt,*curPt])
                prevAngle = (curAng + np.pi) % (np.pi * 2)
                prevPt = tuple(curPt)
                break
    line = np.array(line)[...,:2]
    
    # trim the result points
    trimLine = []
    prevAng = np.pi
    ptr = 0
    while ptr < len(line):
        nextPtr = ptr + 1
        curPt = tuple(line[ptr])
        tarPt = tuple(line[(ptr+1) % len(line)])
        for i in range(len(distList)):
            for j in range(len(ptNeigh[curPt][i])):
                if tuple(ptNeigh[curPt][i][j][:2].astype(int)) == tarPt:
                    curAng = ptNeigh[curPt][i][j][2]
                    break
        if curAng != prevAng:
            trimLine.append(line[ptr])
        prevAng = curAng
        ptr += 1
    trimLine = np.array(list(map(lambda x: liAdd(x, [minX,minY]),trimLine)))
    
    return trimLine