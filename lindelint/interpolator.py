from scipy.spatial import Delaunay, KDTree
import numpy as np
import numpy.linalg as la


class Interpolator():

    def __init__(self, points, properties):

        self.points = points
        self.dim = points.shape[1]
        self.n_points = points.shape[0]

        self.properties = properties
        self.dim_properties = properties.shape[1]

        self.delaunay = None

        if self.dim==2:
            if self.n_points==3:
                self.delaunay=Delaunay(self.points)
            elif self.n_points>=4:
                self.delaunay=Delaunay(self.points, qhull_options='QJ')
        elif self.dim==3:
            if self.n_points==4:
                self.delaunay=Delaunay(self.points)
            elif self.n_points>=5:
                self.delaunay=Delaunay(self.points, qhull_options='QJ')


        if self.delaunay is not None:

            del(self.points)
            self.points=self.delaunay.points

            if self.dim==3:

                self.tetrahedron = []
                self.tetrahedron_dict = {}
                self.n_tetrahedrons = 0
                self.tetrahedron_delaunay_index = []

                self.triangle = []
                self.triangle_dict = {}
                self.n_triangles = 0
                self.triangle_what_tetrahedral = []
                self.triangle_interior_tetrahedral_vertex = []
                self.triangle_outer_normal = []

                self.edge = []
                self.edge_dict = {}
                self.n_edges = 0
                self.edge_what_triangles = []

                self.vertex = []
                self.vertex_dict = {}
                self.n_vertices = 0

                for ii, neighbors in enumerate(self.delaunay.neighbors):
                    if -1 in neighbors:
                        self.tetrahedron.append(self.delaunay.simplices[ii])
                        self.tetrahedron_delaunay_index.append(ii)

                self.tetrahedron = np.array(self.tetrahedron)
                self.tetrahedron_delaunay_index = np.array(self.tetrahedron_delaunay_index)
                self.n_tetrahedrons = self.tetrahedron.shape[0]

                self.triangle = self.delaunay.convex_hull
                self.n_triangles = self.triangle.shape[0]

                for triangle in self.triangle:

                    for pair in [[0,1],[0,2],[1,2]]:
                        edge = np.sort([triangle[pair[0]], triangle[pair[1]]]).tolist()
                        if edge not in self.edge:
                            self.edge.append(edge)

                    for ii, tetrahedron in enumerate(self.tetrahedron):
                        if np.all(np.isin(triangle, tetrahedron, assume_unique=True)):
                            self.triangle_what_tetrahedral.append(ii)
                            interior_tetrahedral_vertex=np.setdiff1d(tetrahedron, triangle, assume_unique=True)[0]
                            p0 = self.points[triangle[0]]
                            p1 = self.points[triangle[1]]
                            p2 = self.points[triangle[2]]
                            pint = self.points[interior_tetrahedral_vertex]
                            normal =  np.cross((p1-p0), (p2-p0))
                            normal = normal/np.linalg.norm(normal)
                            if np.dot((pint-p0),normal)>0.0:
                                normal = -normal
                            self.triangle_interior_tetrahedral_vertex.append(interior_tetrahedral_vertex)
                            self.triangle_outer_normal.append(normal)
                            break

                self.triangle_what_tetrahedral = np.array(self.triangle_what_tetrahedral)
                self.triangle_interior_tetrahedral_vertex = np.array(self.triangle_interior_tetrahedral_vertex)
                self.triangle_outer_normal = np.array(self.triangle_outer_normal)

                self.edge = np.array(self.edge)
                self.n_edges = self.edge.shape[0]

                for edge in self.edge:
                    aux=[]
                    for ii, triangle in enumerate(self.triangle):
                        if np.all(np.isin(edge, triangle)):
                            aux.append(ii)
                            if len(aux)==2:
                                break
                    self.edge_what_triangles.append(aux)

                self.vertex = np.unique(self.edge)
                self.vertices_kdtree = KDTree(self.points[self.vertex])
                self.n_vertices = self.vertex.shape[0]
                self.edge_what_triangles = np.array(self.edge_what_triangles)

                for index, tetrahedron in enumerate(self.tetrahedron):
                    self.tetrahedron_dict[tuple(tetrahedron)]=index

                for index, triangle in enumerate(self.triangle):
                    self.triangle_dict[tuple(triangle)]=index

                for index, edge in enumerate(self.edge):
                    self.edge_dict[tuple(edge)]=index

                for index, vertex in enumerate(self.vertex):
                    self.edge_dict[vertex]=index

            if self.dim==2:

                self.triangle = []
                self.triangle_dict = {}
                self.n_triangles = 0
                self.triangle_delaunay_index = []

                self.edge = []
                self.edge_dict = {}
                self.n_edges = 0
                self.edge_what_triangle = []
                self.edge_opposite_triangle_vertex = []
                self.edge_outer_normal = []

                self.vertex = []
                self.vertex_dict = {}
                self.n_vertices = 0

                for ii, neighbors in enumerate(self.delaunay.neighbors):
                    if -1 in neighbors:
                        self.triangle.append(self.delaunay.simplices[ii])
                        self.triangle_delaunay_index.append(ii)

                self.triangle = np.array(self.triangle)
                self.triangle_delaunay_index = np.array(self.triangle_delaunay_index)
                self.n_triangles = self.triangle.shape[0]

                self.edge = self.delaunay.convex_hull
                self.n_edges = self.edge.shape[0]

                for edge in self.edge:
                    for ii, triangle in enumerate(self.triangle):
                        if np.all(np.isin(edge, triangle)):
                            self.edge_what_triangle.append(ii)
                            opposite_triangle_vertex = np.setdiff1d(triangle, edge, assume_unique=True)[0]
                            self.edge_opposite_triangle_vertex.append(opposite_triangle_vertex)
                            p0 = self.points[edge[0]]
                            p1 = self.points[edge[1]]
                            pint = self.points[opposite_triangle_vertex]
                            p01 = p1-p0
                            p01 = p01/np.linalg.norm(p01)
                            normal = np.array([p01[1], -p01[0]])
                            if np.dot((pint-p0),normal)>0.0:
                                normal = -normal
                            self.edge_outer_normal.append(normal)
                            break

                self.edge_what_triangle = np.array(self.edge_what_triangle)
                self.edge_opposite_triangle_vertex = np.array(self.edge_opposite_triangle_vertex)
                self.edge_outer_normal = np.array(self.edge_outer_normal)

                self.vertex = np.unique(self.edge)
                self.vertices_kdtree = KDTree(self.points[self.vertex])
                self.n_vertices = self.vertex.shape[0]

                for index, triangle in enumerate(self.triangle):
                    self.triangle_dict[tuple(triangle)]=index

                for index, edge in enumerate(self.edge):
                    self.edge_dict[tuple(edge)]=index

                for index, vertex in enumerate(self.vertex):
                    self.edge_dict[vertex]=index


    def do_your_thing(self, points):

        if self.dim==2:
            return self._do_your_thing_2D(points)
        elif self.dim==3:
            return self._do_your_thing_3D(points)
        else:
            raise ValueError('LinDelInt only works in a 2D or 3D space.')


    def _do_your_thing_3D(self, points):

        n_points = points.shape[0]

        properties = np.zeros([n_points, self.dim_properties])

        done = np.zeros([n_points], dtype=bool)


        ## Points inside the convex hull 3D

        simplex_of_point = self.delaunay.find_simplex(points)
        points_of_simplex = {ii:[] for ii in range(self.delaunay.nsimplex)}
        points_of_simplex[-1] = []

        for ii,jj in enumerate(simplex_of_point):
            points_of_simplex[jj].append(ii)

        for ii in range(self.delaunay.nsimplex):

            X = self.delaunay.transform[ii, :3]
            Z = self.delaunay.transform[ii, 3]

            for jj in points_of_simplex[ii]:
                
                point = points[jj]
                Y = point - Z
                b = X.dot(np.transpose(Y))
                bcoords = np.concatenate([b, [1 - b.sum()]])

                properties[jj] = np.dot(bcoords, self.properties[self.delaunay.simplices[ii]])

                done[jj] = True

        del(simplex_of_point, points_of_simplex)


        ## Points in front of triangles 3D

        for ii, triangle in enumerate(self.triangle):

            triangle_0 = triangle[0]
            triangle_1 = triangle[1]
            triangle_2 = triangle[2]

            outer_normal = self.triangle_outer_normal[ii]

            p0 = self.points[triangle_0]
            p1 = self.points[triangle_1]
            p2 = self.points[triangle_2]

            v01 = p1-p0
            d01 = np.linalg.norm(v01)
            b01 = v01/d01

            v02 = p2-p0
            b02 = v02 - np.dot(v02, b01)*b01 
            b02 = b02/np.linalg.norm(b02)

            v01_in_b = [np.dot(v01, b01), 0.0]
            v02_in_b = [np.dot(v02, b01), np.dot(v02, b02)]

            aux_vertices = np.array([[0.0, 0.0], v01_in_b, v02_in_b], dtype=float)

            X = (np.array(aux_vertices[:-1])-aux_vertices[-1]).T
            Y = la.inv(X)
            del(X)

            for jj in range(n_points):
                if not done[jj]:
                    point0 = points[jj]-p0
                    if np.dot(point0, outer_normal)>=0.0:

                        point_in_b = [np.dot(point0, b01), np.dot(point0, b02)]

                        bcoords = np.dot(Y, np.array(point_in_b)-aux_vertices[-1])
                        bcoords.resize(3)
                        bcoords[-1] = 1-bcoords.sum()
                        
                        if (1.0>=bcoords[0]>=0.0):
                            if (1.0>=bcoords[1]>=0.0):
                                if (1.0>=bcoords[2]>=0.0):
                                    done[jj]=True
                                    properties[jj]= bcoords[0]*self.properties[triangle_0]+\
                                                    bcoords[1]*self.properties[triangle_1]+\
                                                    bcoords[2]*self.properties[triangle_2]


        ## Points in front of edges 3D

        for ii, edge in enumerate(self.edge):

            edge_0 = edge[0]
            edge_1 = edge[1]

            p0 = self.points[edge_0]
            p1 = self.points[edge_1]

            v01 = p1-p0
            d01 = np.linalg.norm(v01)
            u01 = v01/d01

            triangle_0, triangle_1 = self.edge_what_triangles[ii]

            oposite_0 = np.setdiff1d(self.triangle[triangle_0], edge, assume_unique=True)[0]
            outer_normal_0 = self.triangle_outer_normal[triangle_0]
            outer_parallel_0 = np.cross(outer_normal_0, u01)
            outer_parallel_0 = outer_parallel_0/np.linalg.norm(outer_parallel_0)
            if np.dot(self.points[oposite_0]-p0, outer_parallel_0)>0.0:
                outer_parallel_0 = -outer_parallel_0

            oposite_1 = np.setdiff1d(self.triangle[triangle_1], edge, assume_unique=True)[0]
            outer_normal_1 = self.triangle_outer_normal[triangle_1]
            outer_parallel_1 = np.cross(outer_normal_1, u01)
            outer_parallel_1 = outer_parallel_1/np.linalg.norm(outer_parallel_1)
            if np.dot(self.points[oposite_1]-p0, outer_parallel_1)>0.0:
                outer_parallel_1 = -outer_parallel_1

            for jj in range(n_points):
                if not done[jj]:
                    point0 = points[jj]-p0
                    if np.dot(point0, outer_parallel_0)>=0.0:
                        if np.dot(point0, outer_parallel_1)>=0.0:
                            f = np.dot(point0, u01)/d01
                            if f>=0.0 and f<=1.0:
                                done[jj]=True
                                properties[jj]= (1.0-f)*self.properties[edge_0]+f*self.properties[edge_1]


        ## Points in corners 3D

        for jj in range(n_points):
            if not done[jj]:
                _, neighbor = self.vertices_kdtree.query(points[jj])
                properties[jj] = self.properties[self.vertex[neighbor]]
                done[jj]=True


        return properties


    def _do_your_thing_2D(self, points):


        n_points = points.shape[0]

        properties = np.zeros([n_points, self.dim_properties])

        done = np.zeros([n_points], dtype=bool)


        ## Points inside the convex hull 2D

        simplex_of_point = self.delaunay.find_simplex(points)
        points_of_simplex = {ii:[] for ii in range(self.delaunay.nsimplex)}
        points_of_simplex[-1] = []

        for ii,jj in enumerate(simplex_of_point):
            points_of_simplex[jj].append(ii)

        for ii in range(self.delaunay.nsimplex):

            X = self.delaunay.transform[ii, :2]
            Z = self.delaunay.transform[ii, 2]

            for jj in points_of_simplex[ii]:
                
                point = points[jj]
                Y = point - Z
                b = X.dot(np.transpose(Y))
                bcoords = np.concatenate([b, [1 - b.sum()]])

                properties[jj] = np.dot(bcoords, self.properties[self.delaunay.simplices[ii]])

                done[jj] = True

        del(simplex_of_point, points_of_simplex)


        ## Points in front of edges 2D

        for ii, edge in enumerate(self.edge):

            edge_0 = edge[0]
            edge_1 = edge[1]

            p0 = self.points[edge_0]
            p1 = self.points[edge_1]

            v01 = p1-p0
            d01 = np.linalg.norm(v01)
            u01 = v01/d01

            outer_normal = self.edge_outer_normal[ii]

            for jj in range(n_points):
                if not done[jj]:
                    point0 = points[jj]-p0
                    if np.dot(point0, outer_normal)>=0.0:
                        f = np.dot(point0, u01)/d01
                        if f>=0.0 and f<=1.0:
                            done[jj]=True
                            properties[jj]= (1.0-f)*self.properties[edge_0]+f*self.properties[edge_1]


        ## Points in corners 2D

        for jj in range(n_points):
            if not done[jj]:
                _, neighbor = self.vertices_kdtree.query(points[jj])
                properties[jj] = self.properties[self.vertex[neighbor]]
                done[jj]=True


        return properties


