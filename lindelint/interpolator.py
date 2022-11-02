from scipy.spatial import Delaunay, KDTree
import numpy as np

class Interpolator():

    def __init__(self, points, properties):

        self.points = points
        self.properties = properties

        self.dim_points = points.shape[1]
        self.dim_properties = properties.shape[1]

        self.delaunay = Delaunay(self.points)
        self._kdtree = KDTree(self.points)
        #self.delaunay.barycenters = self.delaunay.points[self.delaunay.simplices].mean(axis=1)

    def do_your_thing(self, points):

        n_points = points.shape[0]

        properties = np.zeros([n_points, self.dim_properties])

        for point_index, point in enumerate(points):

            simplex_index = self.delaunay.find_simplex(point)

            if simplex_index!=-1:

                X = self.delaunay.transform[simplex_index, :self.delaunay.ndim]
                Y = point - self.delaunay.transform[simplex_index, self.delaunay.ndim]
                b = X.dot(np.transpose(Y))
                bcoords = np.concatenate([b, [1 - b.sum()]])

                properties[point_index] = np.dot(bcoords, self.properties[self.delaunay.simplices[simplex_index]])

            else:

                pass

        return properties

