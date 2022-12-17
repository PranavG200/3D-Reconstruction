import numpy as np
import pdb

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """
    """
    STUDENT CODE BEGINS
    """
    flow = np.zeros((2))
    
    A=[]
    B=[]
    for i in range(size):
      for j in range(size):
        Cornersx = x-2+i <0 or x-2 +i>=Ix.shape[1]
        Cornersy = y-2+j <0 or y-2 +j>=Ix.shape[0]
        if (Cornersy or Cornersx):
          continue
        else:
          A.append([Ix[y-2+j,x-2+i],Iy[y-2+j, x-2+i]])
          B.append(-It[y-2+j,x-2+i])

    A = np.array(A)
    B = np.array(B)

    flow[0], flow[1]= np.linalg.lstsq(A, B, rcond=None)[0]
    u, s, vh = np.linalg.svd(A)
    conf = np.min(s)

    """
    STUDENT CODE ENDS
    """
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

    

