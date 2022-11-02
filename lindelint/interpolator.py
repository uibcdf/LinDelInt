from scipy.spatial import Delaunay, KDTree
import numpy as np

class Interpolator():

    def __init__(self, points, properties):

        self.points = points
        self.properties = properties

        self.dim_points = points.shape[1]
        self.dim_properties = properties.shape[1]

        self.delaunay = Delaunay(self.points)

        self._kdtree = KDTree(points)

        self._convex_hull_simplices = {}

        remain = list(self.delaunay.convex_hull)

        for ii, neighbors in enumerate(self.delaunay.neighbors):
            if -1 in neighbors:
                faces=[]
                aux_list=[]
                simplex=self.delaunay.simplices[ii]
                for face in remain:
                    if np.all(np.isin(face, simplex)):
                        faces.append(face)
                    else:
                        aux_list.append(face)
                self._convex_hull_simplices[ii]=faces
                remain=aux_list

    def do_your_thing(self, points):

        return self._do_your_thing_2D(points)

    def _do_your_thing_3D(self, points):

        n_points = points.shape[0]

        properties = np.zeros([n_points, self.dim_properties])

        simplex_of_point = self.delaunay.find_simplex(points)

        points_of_simplex = {ii:[] for ii in range(self.delaunay.nsimplex)}
        points_of_simplex[-1] = []

        for ii,jj in enumerate(simplex_of_point):
            points_of_simplex[jj].append(ii)

        del(simplex_of_point)


        for simplex_index in range(self.delaunay.nsimplex):

            X = self.delaunay.transform[simplex_index, :self.delaunay.ndim]
            Z = self.delaunay.transform[simplex_index, self.delaunay.ndim]

            for point_index in points_of_simplex[simplex_index]:

                point = points[point_index]
                Y = point - Z
                b = X.dot(np.transpose(Y))
                bcoords = np.concatenate([b, [1 - b.sum()]])

                properties[point_index] = np.dot(bcoords, self.properties[self.delaunay.simplices[simplex_index]])


        remain = points_of_simplex[-1]

        for simplex_index, faces in self._convex_hull_simplices.items():

            simplex = self.delaunay.simplices[simplex_index]

            X = self.delaunay.transform[simplex_index, :self.delaunay.ndim]
            Z = self.delaunay.transform[simplex_index, self.delaunay.ndim]

            for face in faces:

                aux_list = []

                not_in_face = np.where(np.isin(simplex,face)==False)[0]

                for point_index in remain:

                    point = points[point_index]
                    Y = point - Z
                    b = X.dot(np.transpose(Y))
                    bcoords = np.concatenate([b, [1 - b.sum()]])

                    if (bcoords[not_in_face]<0):

                        if self.delaunay.ndim==2:

                            p0 = self.delaunay.points[face[0]]
                            p1 = self.delaunay.points[face[1]]
                            v01 = p1-p0
                            d01 = np.linalg.norm(v01)
                            u01 = v01/d01

                            f = np.dot(point-p0, u01)

                            if f<0:
                                f=0
                            elif f>1:
                                f=1

                            properties[point_index]= (1.0-f)*self.properties[face[0]]+f*self.properties[face[1]]

                    else:

                        aux_list.append(point_index)

                remain=aux_list
                remain=[]

        return properties

    def _do_your_thing_2D(self, points):

        n_points = points.shape[0]

        properties = np.zeros([n_points, self.dim_properties])

        simplex_of_point = self.delaunay.find_simplex(points)

        points_of_simplex = {ii:[] for ii in range(self.delaunay.nsimplex)}
        points_of_simplex[-1] = []

        for ii,jj in enumerate(simplex_of_point):
            points_of_simplex[jj].append(ii)

        del(simplex_of_point)


        for simplex_index in range(self.delaunay.nsimplex):

            X = self.delaunay.transform[simplex_index, :2]
            Z = self.delaunay.transform[simplex_index, 2]

            for point_index in points_of_simplex[simplex_index]:

                point = points[point_index]
                Y = point - Z
                b = X.dot(np.transpose(Y))
                bcoords = np.concatenate([b, [1 - b.sum()]])

                properties[point_index] = np.dot(bcoords, self.properties[self.delaunay.simplices[simplex_index]])


        remain = points_of_simplex[-1]

        for simplex_index, faces in self._convex_hull_simplices.items():

            simplex = self.delaunay.simplices[simplex_index]

            X = self.delaunay.transform[simplex_index, :2]
            Z = self.delaunay.transform[simplex_index, 2]

            for face in faces:

                aux_list = []

                not_in_face = np.where(np.isin(simplex,face)==False)[0]
                in_face = np.where(np.isin(simplex,face)==True)

                p0 = self.delaunay.points[face[0]]
                p1 = self.delaunay.points[face[1]]

                v01 = p1-p0
                d01 = np.linalg.norm(v01)
                u01 = (p1-p0)/d01

                for point_index in remain:

                    point = points[point_index]
                    Y = point - Z
                    b = X.dot(np.transpose(Y))
                    bcoords = np.concatenate([b, [1 - b.sum()]])

                    if (bcoords[not_in_face]<=0):

                        if self.delaunay.ndim==2:

                            f = np.dot(point-p0, u01)/d01

                            if f>=0.0 and f<=1.0:
                                properties[point_index]= (1.0-f)*self.properties[face[0]]+f*self.properties[face[1]]
                            else:
                                aux_list.append(point_index)

                    else:

                        aux_list.append(point_index)

                remain=aux_list

        for point_index in remain:

            _, neighbor = self._kdtree.query(points[point_index])
            properties[point_index] = self.properties[neighbor]

        return properties

