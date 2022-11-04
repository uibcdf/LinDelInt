from scipy.spatial import Delaunay, KDTree
import numpy as np
import numpy.linalg as la

class Interpolator():

    def __init__(self, points, properties):

        self.points = points
        self.properties = properties

        self.n_points = points.shape[0]
        self.dim_points = points.shape[1]
        self.dim_properties = properties.shape[1]

        self.delaunay = None
        self._kdtree = None
        self._convex_hull_simplices = {}
        self._convex_hull_faces = []
        self._convex_hull_edges = []
        self._convex_hull_points = []
        self._faces_with_edge = {}

        if self.dim_points==2:
            if self.n_points==3:
                self.delaunay = Delaunay(self.points)
            elif self.n_points>=4:
                self.delaunay = Delaunay(self.points, qhull_options='QJ')
        elif self.dim_points==3:
            if self.n_points==4:
                self.delaunay = Delaunay(self.points)
            elif self.n_points>=5:
                self.delaunay = Delaunay(self.points, qhull_options='QJ')

        self._kdtree = KDTree(points)

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

        if self.dim_points==2:
            for ii, edges in self._convex_hull_simplices.items():
                for edge in edges:
                    self._convex_hull_edges.append(edge.tolist())
                    for jj in edge:
                        if jj not in self._convex_hull_points:
                            self._convex_hull_points.append(jj)

        if self.dim_points==3:
            for ii, faces in self._convex_hull_simplices.items():
                for face in faces:
                    self._convex_hull_faces.append(face.tolist())
                    edge0=np.sort([face[0], face[1]]).tolist()
                    edge1=np.sort([face[0], face[2]]).tolist()
                    edge2=np.sort([face[1], face[2]]).tolist()
                    if edge0 not in self._convex_hull_edges:
                        self._convex_hull_edges.append(edge0)
                        self._faces_with_edge[tuple(edge0)]=[]
                    if edge1 not in self._convex_hull_edges:
                        self._convex_hull_edges.append(edge1)
                        self._faces_with_edge[tuple(edge1)]=[]
                    if edge2 not in self._convex_hull_edges:
                        self._convex_hull_edges.append(edge2)
                        self._faces_with_edge[tuple(edge2)]=[]
                    self._faces_with_edge[tuple(edge0)].append(face)
                    self._faces_with_edge[tuple(edge1)].append(face)
                    self._faces_with_edge[tuple(edge2)].append(face)
            for edge in self._convex_hull_edges:
                for jj in edge:
                    if jj not in self._convex_hull_points:
                        self._convex_hull_points.append(jj)

    def do_your_thing(self, points):

        if self.dim_points==2:
            return self._do_your_thing_2D(points)
        elif self.dim_points==3:
            return self._do_your_thing_3D(points)
        else:
            raise ValueError('LinDelInt only works in a 2D or 3D space.')


    def _do_your_thing_3D(self, points):

        n_points = points.shape[0]

        properties = np.zeros([n_points, self.dim_properties])

        simplex_of_point = self.delaunay.find_simplex(points)

        points_of_simplex = {ii:[] for ii in range(self.delaunay.nsimplex)}
        points_of_simplex[-1] = []

        for ii,jj in enumerate(simplex_of_point):
            points_of_simplex[jj].append(ii)

        del(simplex_of_point)

        va=[]

        for simplex_index in range(self.delaunay.nsimplex):

            X = self.delaunay.transform[simplex_index, :3]
            Z = self.delaunay.transform[simplex_index, 3]

            for point_index in points_of_simplex[simplex_index]:

                point = points[point_index]
                Y = point - Z
                b = X.dot(np.transpose(Y))
                bcoords = np.concatenate([b, [1 - b.sum()]])

                properties[point_index] = np.dot(bcoords, self.properties[self.delaunay.simplices[simplex_index]])


        remain = points_of_simplex[-1]

        print(len(remain))

        afuera={tuple(face):[] for face in self._convex_hull_faces}

        for simplex_index, faces in self._convex_hull_simplices.items():

            simplex = self.delaunay.simplices[simplex_index]

            X = self.delaunay.transform[simplex_index, :3]
            Z = self.delaunay.transform[simplex_index, 3]

            for face in faces:

                aux_list = []

                not_in_face = np.where(np.isin(simplex,face)==False)[0]
                in_face = np.where(np.isin(simplex,face)==True)


                p0 = self.delaunay.points[face[0]]
                p1 = self.delaunay.points[face[1]]
                p2 = self.delaunay.points[face[2]]

                v01 = p1-p0
                d01 = np.linalg.norm(v01)
                b01 = v01/d01

                v02 = p2-p0
                b02 = v02 - np.dot(v02, b01)*b01 
                b02 = b02/np.linalg.norm(b02)

                v01_in_b = [np.dot(v01, b01), 0.0]
                v02_in_b = [np.dot(v02, b01), np.dot(v02, b02)]

                aux_vertices = np.array([[0.0, 0.0], v01_in_b, v02_in_b], dtype=float)

                XX = (np.array(aux_vertices[:-1])-aux_vertices[-1]).T
                YY = la.inv(XX)
                del(XX)

                for point_index in remain:

                    point = points[point_index]
                    Y = point - Z
                    b = X.dot(np.transpose(Y))
                    bcoords = np.concatenate([b, [1 - b.sum()]])

                    if (bcoords[not_in_face]<=0):


                        point0 = point-p0
                        point_in_b = [np.dot(point0, b01), np.dot(point0, b02)]

                        bbcoords = np.dot(YY, np.array(point_in_b)-aux_vertices[-1])
                        bbcoords.resize(3)
                        bbcoords[-1] = 1-bbcoords.sum()
                        
                        flag = True

                        if (1.0>=bbcoords[0]>=0.0):
                            if (1.0>=bbcoords[1]>=0.0):
                                if (1.0>=bbcoords[2]>=0.0):
                                    properties[point_index]= bbcoords[0]*self.properties[face[0]]+\
                                                             bbcoords[1]*self.properties[face[1]]+\
                                                             bbcoords[2]*self.properties[face[2]]
                                    flag = False

                        if flag:
                            aux_list.append(point_index)
                            afuera[tuple(face)].append(point_index)

                    else:

                        aux_list.append(point_index)

                remain=aux_list

        print(len(remain))


        kdtree = KDTree(points[self._convex_hull_points])
        neighbors_convex_hull_point = {ii:[] for ii in self._convex_hull_points}
        neighbors={}
        for point_index in remain:
            _, ii = kdtree.query(points[point_index])
            jj = self._convex_hull_points[ii]
            neighbors_convex_hull_point[jj].append(point_index)
            neighbors[point_index]=jj

        for edge in self._convex_hull_edges:

            aux_neighbors=neighbors_convex_hull_point[edge[0]]+neighbors_convex_hull_point[edge[1]]
            aux_afuera=[]
            for face in self._faces_with_edge[tuple(edge)]:
                aux_afuera+=afuera[tuple(face)]
            aux_afuera = np.unique(aux_afuera)

            aux_list = np.intersect1d(aux_neighbors, aux_afuera)
            aux_list = np.intersect1d(aux_list, remain)

            p0 = self.delaunay.points[edge[0]]
            p1 = self.delaunay.points[edge[1]]

            v01 = p1-p0
            d01 = np.linalg.norm(v01)
            u01 = (p1-p0)/d01

            for point_index in aux_list:

                point = points[point_index]

                f = np.dot(point-p0, u01)/d01

                if f>=0.0 and f<=1.0:

                    properties[point_index]= (1.0-f)*self.properties[edge[0]]+f*self.properties[edge[1]]
                    remain.remove(point_index)


        print(len(remain))

        for point_index in remain:

            properties[point_index] = self.properties[neighbors[point_index]]
            va.append(point_index)

        #return properties
        return va

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

