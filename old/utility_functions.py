import numpy as np
import scipy.linalg as la
from geomdl import NURBS
from geomdl.fitting import interpolate_curve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from geomdl import BSpline
from geomdl import utilities
from scipy.spatial.transform import Rotation as R

class Utility_functions:
    def __init__(self):
        pass

    def surface_curvature(self,hessian):
        '''
        This function finds the eigenvalues and eigenvectors of the hessian
        matrix considering only the z component; it works by creating a square
        matrix for every point, then evaluating the eigenvalues and eigenvectors of
        that matrix, then storing eigenvalues as principal curvatures and eigenvectors
        as principal directions
        :param
        hessian:  The hessian matrix of the NURBS surface
        :return:
        k1: first principal curvature
        k2: second principal curvature
        d1: first principal direction
        d2: second principal direction
        kg: gaussian curvature
        km: mean curvature
        '''
        hess_uu = hessian[0,...]
        hess_uv = hessian[1,...]
        hess_vu = hessian[2,...]
        hess_vv = hessian[3,...]
        # Gaussian and mean curvature initialization
        kg = np.zeros((len(hess_vv[:,0]),))
        km = np.zeros((len(hess_vv[:,0]),))
        k1,k2 = [],[]
        d1 = np.zeros((len(hess_vv[:,0]),2))
        d2 = np.zeros((len(hess_vv[:,0]),2))
        # Hessian matrix components along z
        Hessian_z = np.zeros((2,2))
        for u in range(len(hess_vv[:,0])):
            Hessian_z[0,0] = hess_uu[u,2]
            Hessian_z[0,1] = hess_uv[u,2]
            Hessian_z[1,0] = hess_vu[u,2]
            Hessian_z[1,1] = hess_vv[u,2]
            V,_ = np.linalg.eig(Hessian_z)
            ''' Gaussian curvature'''
            kg[u] = V[0]*V[1]
            '''Mean curvature '''
            km[u] = (V[0]+V[1])/2*-1
            '''Principal curvatures '''
            index_max = np.argmax([np.abs(V[0]),np.abs(V[1])])
            index_min = np.argmin([np.abs(V[0]),np.abs(V[1])])
            k1.append(V[index_max])
            k2.append(V[index_min])
            '''Eigenvectors '''
            d1[u, :] = _[:,index_max]
            d2[u, :] = _[:,index_min]

        return [k1,k2,d1,d2,kg,km]

    def invalid_data_check(self,vector_uv):
        '''
        This function checks for out of scope parametric variables.
        It works by assigning to the output vector only the points whose
        u and v parametric coordinates both belong to the interval [0,1]
        Input vector_uv is a 2xn where the 1st row represents the u coordinates
        and the 2nd row represents the v coordinates
        Output processed_vector is a 2 x m where m<=n.'''
        uu = vector_uv[0,:]
        vv = vector_uv[1,:]
        processed_data = []
        for i in range(int(np.size(uu))):
            if (uu[i]*vv[i]>=0 and uu[i]<=1 and vv[i]<=1 and uu[i]>0 and vv[i]>0):
                processed_data.append(vector_uv[:,i])
        return np.asarray(processed_data)


    def axis_magnitude(self,k1,k2,h,r):
        ''' Function that computes the magnitude of the axis of the contact area
        INPUTS
        k1: first principal curvature
        k2: second principal curvature
        r: radius of the tool
        h: Normal Force / compliance of the tool

        '''
        if (np.round(k1,3)!=0 and np.round(k2,3)!=0):
            rho = np.asarray([1/k1, 1/k2])
            w = np.sqrt(np.square(rho) - np.square(abs(rho) - np.asarray([h,h])))
            if w[0] > r:
              w[0] = r
            if w[1] > r:
              w[1] = r
        elif np.round(k1,3)==0:
            rho = 1/k2
            w = np.sqrt(np.square(rho) - np.square(abs(rho) - h) )
            if w>r:
                w = r
        elif np.round(k2,3)==0:
            rho = 1/k1
            w = np.sqrt(np.square(rho) - np.square(abs(rho) - h) )
            if w>r:
                w = r
        return w

    def oriented_elliptic_contact(self,a,b,d1,d2,Su,Sv,p,n_points_ellipse):
            '''
            This function takes as input the semi major and semi minor axis of the
            ellipse as well as the eigenvalues  and eigenvectors and the
            derivative of the surface along u and the derivative of the surface along
            v to eventually compute the contact area and the orientation of the
            ellipse.
            INPUTS:
            a: semi major axis computed at point of evaluation
            b: semi minor axis computed at point of evaluation
            d1: 1st principal direction at point of evaluation
            d2: 2nd principal direction at point of evaluation
            Su: derivative of the surface along u direction at point of evalutation
            Sv: derivative of the surface along v direction at point of evalutation
            p: point of evaluation

            OUTPUTS:
            X,Y,Z: POINTS OF THE ELLIPSE THAT ARE ROTATED AND TRANSLATED according to
            surface geometry; X,Y and Z are row vectors
            index_max: this is the index of the point of the ellipse with the max Y
            index_min: this is the index of the point of the ellipse with the min Y
            '''
            theta = np.linspace(0, 360,n_points_ellipse)
            x = a * np.cos(np.deg2rad(theta)) # computed in frame dir1, dir2 (eigenvector)
            y = b * np.sin(np.deg2rad(theta)) # computed in frame dir1, dir2 (eigenvector)
            z =  np.zeros((n_points_ellipse,))
            x_y_z =  np.zeros((3,n_points_ellipse))
            dir_1_2_3 =  np.zeros((3,3))
            dir1 = (d1[0] * Su + d1[1] * Sv)
            dir2 = (d2[0] * Su + d2[1] * Sv)
            normal = np.cross(Su,Sv)

            for i in range(n_points_ellipse):
             x_y_z[:,i] = np.asarray([x[i],y[i],z[i]])
            dir_1_2_3[:,0] = dir1
            dir_1_2_3[:,1] = dir2
            dir_1_2_3[:,2] = normal
            Points_ellipse_rotated = -np.matmul(dir_1_2_3, x_y_z)
            Points_ellipse_rotated_and_translated = Points_ellipse_rotated

            for pp in range(n_points_ellipse):
               Points_ellipse_rotated_and_translated[:,pp] = Points_ellipse_rotated_and_translated[:,pp]+ np.asarray(p)
            XX = Points_ellipse_rotated_and_translated[0,:]
            YY = Points_ellipse_rotated_and_translated[1,:]
            ZZ = Points_ellipse_rotated_and_translated[2,:]

            # This for loop is to find the X  associated to the upper and lower boundary point
            for i in range(n_points_ellipse):
                        if YY[i] == np.max(YY):
                            index_max = i
                        if YY[i] == np.min(YY):
                            index_min = i

            return XX,YY,ZZ,index_max, index_min

    def parabolic_contact(self,w,d1,d2,Su,Sv,p,r):
        '''
        This function takes as input the axis of the
        circle as well as the eigenvalues  and eigenvectors and the
        derivative of the surface along u and the derivative of the surface along
        v to eventually compute the contact area and the orientation of the
        parabolic contact.
        INPUTS:
        w:  axis computed at point of evaluation
        d1: 1st principal direction at point of evaluation
        d2: 2nd principal direction at point of evaluation
        Su: derivative of the surface along u direction at point of evalutation
        Sv: derivative of the surface along v direction at point of evalutation
        p: point of evaluation

        OUTPUTS:
        X,Y,Z: POINTS OF THE ELLIPSE THAT ARE ROTATED AND TRANSLATED according to
        surface geometry; X,Y and Z are row vectors
        index_max: this is the index of the point of the parabolic contact with the max Y
        index_min: this is the index of the point of the parabolic contact  with the min Y
        '''
        alpha = np.arcsin(w / r)
        theta_to_eval = list(np.arange(0,alpha, 0.01)) + list(np.arange(np.math.pi - alpha, alpha + np.math.pi, 0.01)) +  list(np.arange((2 * np.math.pi - alpha), 2 * np.math.pi, 0.01))
        x = r * np.cos(theta_to_eval)
        y = r * np.sin(theta_to_eval)
        z = np.zeros((len(theta_to_eval),))
        x_y_z = np.zeros((3, len(theta_to_eval)))
        dir_1_2_3 = np.zeros((3, 3))
        dir1 = (d1[0] * Su + d1[1] * Sv)
        dir2 = (d2[0] * Su + d2[1] * Sv)
        normal = np.cross(Su, Sv)

        for i in range(len(theta_to_eval)):
            x_y_z[:, i] = np.asarray([x[i], y[i], z[i]])
        dir_1_2_3[:, 0] = dir1
        dir_1_2_3[:, 1] = dir2
        dir_1_2_3[:, 2] = normal
        Points_parabolic_contact = -np.matmul(dir_1_2_3, x_y_z)
        Points_parabolic_contact_rotated_and_translated = Points_parabolic_contact

        for pp in range(len(theta_to_eval)):
            Points_parabolic_contact_rotated_and_translated[:, pp] = Points_parabolic_contact_rotated_and_translated[:, pp] + np.asarray(p)
        XX = Points_parabolic_contact_rotated_and_translated[0, :]
        YY = Points_parabolic_contact_rotated_and_translated[1, :]
        ZZ = Points_parabolic_contact_rotated_and_translated[2, :]

        # This for loop is to find the X  associated to the upper and lower boundary point
        for i in range(len(theta_to_eval)):
            if YY[i] == np.max(YY):
                index_max = i
            if YY[i] == np.min(YY):
                index_min = i

        return XX,YY,ZZ,index_max, index_min

    def NURBS_Surface_building(self,pnts,knots, coef=0.028):
        '''
        INPUTS:
        pnts : control points of the surface
        knots : knot vectors
        coef : the scaling coefficient (enter values in mm and divide by coef to get values in meter)

        OUTPUTS:
        surf :  NURBS surface instance'''

        print('Building the NURBS Surface')
        ''' Scaling the control points '''
        for i in range(len(pnts)):
            for j in range(len(pnts[0]) - 1):
                pnts[i][j] = pnts[i][j] * coef
        u_pnts = []
        v_pnts = []
        for ii in range(len(pnts)):
            u_pnts.append(pnts[ii][0])
        for jj in range(len(pnts)):
            v_pnts.append(pnts[jj][1])

        ''' Create a NURBS surface instance '''
        surf = NURBS.Surface()
        # Set degrees
        surf.degree_u = 2
        surf.degree_v = 2
        # Set control points
        surf.ctrlpts2d = [pnts[0:3], pnts[3:6], pnts[6:9]]
        surf.knotvector_u = knots[0]
        surf.knotvector_v = knots[1]
        surf.sample_size = 40

        return surf


    def Surface_dimensions(self,surf,coef=0.028):
        '''
        INPUTS:
        surf : NURBS surface instance
        coef : the scaling coefficient

        OUTPUTS:
        dim_u : u dimension
        dim_v : v dimension
        surface_size_u : normalized u dimension
        surface_size_v : normalized v dimension
        surface_length : maximum dimension of the surface
        '''
        control_points = np.asarray(surf.ctrlpts2d)
        pnts =[]
        for i in range(control_points.shape[0]):
            for j in range(control_points.shape[1]):
               pnts.append(control_points[i,j])

        ''' surface '''
        u_pnts = []
        v_pnts = []
        for ii in range(len(pnts)):
            u_pnts.append(pnts[ii][0])
        for jj in range(len(pnts)):
            v_pnts.append(pnts[jj][1])
        ''' Compute surface dimensions '''
        surface_size_u = np.max(u_pnts)
        surface_size_v = np.max(v_pnts)
        surface_length = np.max([surface_size_u, surface_size_v])
        print('The size of the surface is:', surface_size_u, ' by ', surface_size_v)
        print('The length of the surface is:', surface_length)

        ''' Normalized dimensions of the surface '''
        dim_u = surface_size_u / coef
        dim_v = surface_size_v / coef
        return dim_u, dim_v, surface_size_u, surface_size_v, surface_length

    def Trajectory_generation(self,surf,surface_length,surface_size_u,surface_size_v,r=0.05,F= 10,kc= 40000,n_points_ellipse = 500,coef=0.028):
        ''' Trajectory generation function

         INPUTS:
         surf: NURBS surface instance
         surface_length: Maximum dimension of the durface
         surface_size_u: U dimension of the surface
         surface_size_v: V dimension of the surface
         r: Radius of the tool
         F: Normal Force
         kc: Compliance of the surface-tool coupling
         n_points_ellipse: Number of points of the ellipse of contact
         coef: Scaling coefficient

         OUTPUTS:
         surfpts: Points of the surface
         initial_curve_eval_points: Initial path evaluation points
         initial_curve_points:  Initial path  points
         initial_path: Initial path curve
         points: Points of the computed trajectory
         points_normal: Normal vector along the trajectory expressed as vector coordinates in the local reference system.
                        Every normal vector has one end localized in the point of the trajectory and the other is stored in points_normal.
         curve_final : Final complete path curve
         '''
        ''' Evaluate the points of the surface'''
        surf.evaluate()
        surfpts = np.array(surf.evalpts)

        ''' Initial isocurve computation'''
        npts = np.round((surface_length / coef) + 1)
        print('Number of points on the initial isocurve: ', int(npts))
        ut = np.linspace(0.05, 0.95, int(npts), dtype=float)    # Parametric evaluation points along U direction:
        vt = np.repeat(0.05, npts)                              # Parametric evaluation points along V direction
        tt = []
        for p in range(int(npts)):
            tt.append((vt[p], ut[p]))
        crv_pnt = surf.evaluate_list(tt)
        initial_path = interpolate_curve(crv_pnt, 2)

        ''' Initial path points'''
        initial_path.evaluate()
        initial_curve_points = np.array(initial_path.evalpts)

        ''' Evaluates the derivative of the curve along the points'''
        points = []
        dpoints = []
        for point in range(int(npts)):
            points.append(initial_path.derivatives(ut[point], order=1)[0])
            dpoints.append(initial_path.derivatives(ut[point], order=1)[1])
        points = np.asarray(points)
        dpoints = np.asarray(dpoints)

        ''' Initial path evaluation points'''
        initial_curve_eval_points = np.asarray(points)

        '''Normalization'''
        f = dpoints
        for i in range(int(npts)):
            norm = np.linalg.norm(dpoints[i, :])
            f[i, :] = dpoints[i, :] / norm

        ''' Defining the derivatives along u and v of the surface and the normal'''
        p1 = []
        Su, Sv = [], []
        hess_uu, hess_vv, hess_uv, hess_vu = [], [], [], []

        for point in range(int(npts)):
            p1.append(surf.derivatives(u=vt[point], v=ut[point], order=2)[0][0])
            # JACOBIAN COMPONENTS
            Sv.append(surf.derivatives(u=vt[point], v=ut[point], order=2)[1][0])  # 1 deriv wrt v
            Su.append(surf.derivatives(u=vt[point], v=ut[point], order=2)[0][1])  # 1 deriv wrt u
            # HESSIAN COMPONENTS
            hess_uu.append(surf.derivatives(u=vt[point], v=ut[point], order=2)[0][2])  # 1 deriv wrt v
            hess_vv.append(surf.derivatives(u=vt[point], v=ut[point], order=2)[2][0])  # 1 deriv wrt u
            hess_uv.append(surf.derivatives(u=vt[point], v=ut[point], order=2)[1][1])  # 1 deriv wrt v
            hess_vu.append(surf.derivatives(u=vt[point], v=ut[point], order=2)[1][1])  # 1 deriv wrt u

        p1 = np.asarray(p1)

        ''' Normalization'''
        Su = np.asarray(Su)
        Sv = np.asarray(Sv)
        for i in range(int(npts)):
            norm_u = np.linalg.norm(Su[i, :])
            norm_v = np.linalg.norm(Sv[i, :])
            Su[i, :] = Su[i, :] / norm_u
            Sv[i, :] = Sv[i, :] / norm_v

        ''' Normal definition'''
        normal = np.cross(Su, Sv)

        ''' Hessian matrix construction '''
        hess_uu = np.asarray(hess_uu)
        hess_vv = np.asarray(hess_vv)
        hess_uv = np.asarray(hess_uv)
        hess_vu = np.asarray(hess_vu)
        d2p = np.empty((4, int(npts), 3))
        d2p[0, :, :] = hess_uu
        d2p[1, :, :] = hess_uv
        d2p[2, :, :] = hess_vu
        d2p[3, :, :] = hess_vv
        d = np.cross(normal, f)
        ''' Compute surface curvature '''
        h = F / kc
        [k1, k2, d1, d2, kg, km] = self.surface_curvature(d2p)
        upper_boundary = np.zeros((3, int(npts)))  # Coordinate of point on upper boundary curve
        lower_boundary = np.zeros((3, int(npts)))  # Coordinate of point on lower boundary curve
        w, a, b = [], [], []
        X, Y, Z = [], [], []
        ''' Contact area evaluation'''
        for u in range(int(npts)):
            w.append(self.axis_magnitude(k1[u], k2[u], h, r))
            if (np.round(kg[u], 3) > 0 and np.round(km[u], 3) > 0) or (np.round(kg[u], 3) > 0 and np.round(km[u],3) < 0):  # contact area is an Ellipse
                a.append(w[u][0])           # semi major axis of ellipse
                b.append(w[u][1])           # semi minor axis of ellipse
                xx, yy, zz, index_max, index_min = self.oriented_elliptic_contact(a[u], b[u], d1[u, :], d2[u, :],Su[u, :], Sv[u, :], p1[u, :],n_points_ellipse)
                X.append(xx)
                Y.append(yy)
                Z.append(zz)
                upper_boundary[:, u] = np.asarray([np.transpose(np.asarray(X))[index_max, u], max(np.transpose(np.asarray(Y))[:, u]),np.transpose(np.asarray(Z))[index_max, u]])
                lower_boundary[:, u] = np.asarray([np.transpose(np.asarray(X))[index_min, u], min(np.transpose(np.asarray(Y))[:, u]),np.transpose(np.asarray(Z))[index_min, u]])

            if (np.round(kg[u], 3) == 0 and np.round(km[u], 3) != 0):
                xx, yy, zz, index_max, index_min = self.parabolic_contact(w[u], d1[u, :], d2[u, :], Su[u, :],Sv[u, :], p1[u, :], r)
                X.append(xx)
                Y.append(yy)
                Z.append(zz)
                upper_boundary[:, u] = np.asarray([np.transpose(np.asarray(X))[index_max, u], max(np.transpose(np.asarray(Y))[:, u]),np.transpose(np.asarray(Z))[index_max, u]])  # The ellipse is defined on the (Su, Sv) plane tangent to the surface so the points of the ellipse do not belong to the surface
                lower_boundary[:, u] = np.asarray([np.transpose(np.asarray(X))[index_min, u], min(np.transpose(np.asarray(Y))[:, u]),np.transpose(np.asarray(Z))[index_min, u]])  # We just used the X and Y coordinated of the points of the ellipse, and then the z coord is found later using nrbeval.
        '''Interpolation of the upper boundary contact points to find the 1st ribbon boundary '''
        upper = []
        for i in range(int(npts)):
            upper.append(upper_boundary[:, i])

        '''Normalizing to get parametric domain in u and v direction'''
        tu = upper_boundary[0, :] / surface_size_u
        tv = upper_boundary[1, :] / surface_size_v
        xxx = []
        for p in range(int(npts)):
            xxx.append([tv[p], tu[p]])
        # The cartesian points of upper bound actually belong to the surface so the Z is actually on the surface.
        upper_boundary = surf.evaluate_list(xxx)
        upper = []
        for i in range(int(npts)):
            upper.append(np.asarray(upper_boundary)[i, :])
        upper_boundary = np.asarray(upper)


        ''' 
        Path generation procedure
        EXPLANATION OF ALGORITHM:
        1- we started with a set of base points P_i_j where i represents the ith base point ( i = 0:npts )in the jth run (here j = 0). 
        2- We found the contact area at each point, and identified the point with
        max and min coord in the y- direction (lower boundary point (b_l) and upper boundary point (b_u))
        3- We found the upper boundary curve S0 by interpolating  the points b_u_i that we now call S_i_j
        4- At each point S_i_j , we found the proper offset, such that the b_l associated to P_i_j+1 touches S_i_j  
        The offset search interval at each point is [0,2*r] in the y direction (so along the vector d)
        We employ the bisection method to find the proper offset that leads to error ( as defined by equation 8 in reference 2 )
        '''

        fig1 = plt.figure(figsize=(10.67, 8), dpi=96)
        ax1 = Axes3D(fig1)

        curvepts = np.array(initial_path.evalpts)
        ax1.plot_trisurf(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], cmap=cm.summer, alpha=0.5)
        ax1.plot(curvepts[:, 0], curvepts[:, 1], curvepts[:, 2], color='brown', linestyle='-', linewidth=1)

        # Plotting of the initial contact ellipses
        for u in range(int(npts)):
            ax1.plot3D(np.transpose(np.asarray(X))[:, u], np.transpose(np.asarray(Y))[:, u],np.transpose(np.asarray(Z))[:, u])

        ''# Variables initialization
        limit_y = np.repeat(surface_size_v, npts)
        error = 1
        tol_y = max(np.transpose(np.asarray(Y))[:, 0]) - min(np.transpose(np.asarray(Y))[:, 0])
        tol = 1e-6
        flag = 0
        u_coord = np.linspace(0.05, 0.95, int(npts))
        trial_upper = np.empty((int(npts), 3, int(npts)))
        lower_bound, upper_bound, midpoint = [], [], []
        u_v = np.zeros((2, int(npts)))
        storage_upper_bound = np.zeros((int(npts), int(npts), 2))
        storage_lower_bound = np.zeros((int(npts), int(npts), 2))
        path_pnts = np.zeros((3, int(npts), int(npts)))
        normal_pnts = np.zeros((3, int(npts), int(npts)))
        XX, YY, ZZ = [], [], []

        upper_boundary = np.transpose(upper_boundary)
        for v in range(int(npts)):
            trial_upper[v, :, :] = upper_boundary
            for u in range(int(npts)):
                if trial_upper[v, 1, u] / surface_size_v < (1 - tol_y):
                    if limit_y[u] - upper_boundary[1, u] > tol_y:  # While the upper boundary of the polishing path has still not reached the ending of the surface continue looping

                        lower_bound.append(upper_boundary[1, u] / round(surface_size_v,4))            # V coordinate of upper boundary point
                        upper_bound.append((upper_boundary[1, u] + 2 * r) / round(surface_size_v,4))  # V coordinate of upper boundary point offset by 2 r; this is the limit of the search interval;
                                                                                                      # In case the 2*r offset leads to a point outisde the surface, the invalid data check done previously
                                                                                                      # will adjust the upper limit to a point belonging to the surface.

                        if upper_bound[-1] > 1:
                            upper_bound[-1] = 1
                        midpoint.append((lower_bound[-1] + upper_bound[-1]) / 2)

                        while error != 0:
                            ut = u_coord[u]
                            vt = midpoint[-1]
                            s = [ut, vt]
                            P = surf.derivatives(u=s[1], v=s[0], order=2)[0][0]

                            '''JACOBIAN COMPONENTS'''
                            SU = np.asarray(surf.derivatives(u=s[1], v=s[0], order=2)[0][1])  # 1 deriv wrt v
                            SV = np.asarray(surf.derivatives(u=s[1], v=s[0], order=2)[1][0])  # 1 deriv wrt u
                            norm_U = np.linalg.norm(SU)
                            norm_V = np.linalg.norm(SV)
                            SU = SU / norm_U
                            SV = SV / norm_V

                            '''HESSIAN COMPONENTS'''
                            HESS_UU = np.asarray(surf.derivatives(u=s[1], v=s[0], order=2)[0][2])  # 1 deriv wrt v
                            HESS_VV = np.asarray(surf.derivatives(u=s[1], v=s[0], order=2)[2][0])  # 1 deriv wrt u
                            HESS_UV = np.asarray(surf.derivatives(u=s[1], v=s[0], order=2)[1][1])  # 1 deriv wrt v
                            HESS_VU = np.asarray(surf.derivatives(u=s[1], v=s[0], order=2)[1][1])  # 1 deriv wrt u

                            ''' Derivatives wrt U and V and normal definition'''
                            NORMAL = np.cross(SU, SV)

                            ''' Hessian matrix construction '''
                            d2p = np.empty((4, 1, 3))
                            d2p[0, :, :] = HESS_UU
                            d2p[1, :, :] = HESS_UV
                            d2p[2, :, :] = HESS_VU
                            d2p[3, :, :] = HESS_VV

                            ''' Normalization'''
                            [k1, k2, d1, d2, kg, km] = self.surface_curvature(d2p)
                            k1 = k1[0]
                            k2 = k2[0]
                            w = self.axis_magnitude(k1, k2, h, r)
                            if (np.round(kg, 3) > 0 and np.round(km, 3) > 0) or (np.round(kg, 3) > 0 and np.round(km,3) < 0):
                                a = w[0]
                                b = w[1]
                                X, Y, Z, index_max, index_min = self.oriented_elliptic_contact(a, b,np.squeeze(d1),np.squeeze(d2), SU,SV, P, 500)
                                upper_boundary_new = np.asarray([X[index_max], max(Y[:]), Z[index_max]])
                                lower_boundary_new = np.asarray([X[index_min], min(Y[:]), Z[index_min]])
                            if np.round(kg, 3) == 0 and np.round(km, 3) != 0:
                                X, Y, Z, index_max, index_min = self.parabolic_contact(w, np.squeeze(d1),np.squeeze(d2), SU, SV, P,r)
                                upper_boundary_new = np.asarray([X[index_max], max(Y[:]), Z[index_max]])

                            error = lower_boundary_new[1] - upper_boundary[1, u]

                            if error < 0:
                                lower_bound[-1] = midpoint[-1]
                                midpoint[-1] = (lower_bound[-1] + upper_bound[-1]) / 2
                            if error > tol:
                                upper_bound[-1] = midpoint[-1]
                                midpoint[-1] = (lower_bound[-1] + upper_bound[-1]) / 2
                            if error <= tol and error >= 0:
                                u_v[:, u] = [upper_boundary_new[0] / surface_size_u, upper_boundary_new[1] / surface_size_v]
                                storage_upper_bound[u, v] = np.asarray([upper_boundary_new[0] / surface_size_u, upper_boundary_new[1] / surface_size_v])
                                storage_lower_bound[u, v] = np.asarray([lower_boundary_new[0] / surface_size_u,lower_boundary_new[1] / surface_size_v])
                                path_pnts[:, u, v] = P
                                normal_pnts[:, u, v] = NORMAL
                                ax1.plot3D(X, Y, Z)
                                XX.append(X)
                                YY.append(Y)
                                ZZ.append(Z)
                                error = 0
                        error = 1
                else:
                    flag = 1

            if flag == 1:  # If used to break outside of the v loop in case boundary is reached
                break

            u_v = self.invalid_data_check(u_v)
            u_v_list = []
            for iii in range(int(npts)):
                u_v_list.append([u_v[iii][1], u_v[iii][0]])

            upper_boundary = surf.evaluate_list(u_v_list)
            S_1 = interpolate_curve(upper_boundary, 2)
            upper_boundary = np.asarray(np.transpose(upper_boundary))
            u_v = np.empty((2, int(npts)))
            lower_bound, upper_bound, midpoint = [], [], []

        path_pnts_processed = []
        points_normal_processed = []
        points = crv_pnt
        points_normal = list(normal)

        for v in range(int(npts)):
            for u in range(int(npts)):
                for i in range(3):
                    if path_pnts[i, u, v] != 0:
                        path_pnts_processed.append([path_pnts[0, u, v], path_pnts[1, u, v], path_pnts[2, u, v]])
                        points_normal_processed.append([normal_pnts[0, u, v], normal_pnts[1, u, v], normal_pnts[2, u, v]])

            points_orig = points
            normal_points_orig = points_normal
            path_pnts_processed_reversed = path_pnts_processed[::-1]
            normal_points_processed_reversed = points_normal_processed[::-1]

            if (v + 1) % 2 == 0:
                points = points_orig + path_pnts_processed
                points_normal = normal_points_orig + points_normal_processed

            else:

                points = points_orig + path_pnts_processed_reversed
                points_normal = normal_points_orig + normal_points_processed_reversed

            ax1.plot(np.asarray(points)[:, 0], np.asarray(points)[:, 1], np.asarray(points)[:, 2], color='red',linestyle='-', linewidth=3)
            ''' Finding the path interpolating the points found with the bisection method '''
            curve_final = BSpline.Curve()
            curve_final.degree = 2
            curve_final.ctrlpts = points
            curve_final.knotvector = utilities.generate_knot_vector(curve_final.degree, len(curve_final.ctrlpts))
            curve_final.delta = 0.05
            path_pnts_processed = []
            points_normal_processed = []

        ax1.set_xlabel('$u$')
        ax1.set_ylabel('$v$')
        fig1.suptitle('Polishing path')
        ax1.set_zlim3d(top=surface_length)
        plt.show()


        return surfpts, initial_curve_eval_points, initial_curve_points, initial_path, np.asarray(points), np.asarray(points_normal), curve_final

    def Rotation_Matrix_from_Normal_Vector(self,points,points_normal):
        '''
        INPUTS:
        points : array eith the polishing path points coordinates
        points_normal: array of coordinates of the normal vector in the local reference system with shape (n_points,3)

        OUTPUTS:
        Rotation_matrix: Rotation matrix  in every point of the trajectory defined to have the z axis of the end effector aligned with the normal vector
        '''
        npoints = points.shape[0]
        ''' Initialization'''
        tx = np.zeros((npoints,3))
        ty = np.zeros((npoints,3))
        tz = np.zeros((npoints,3))
        Rotation_matrix = np.zeros((npoints,3,3))

        tz[:,0] = -(points_normal[:, 0] - points[:, 0])
        tz[:,1] = -(points_normal[:, 1] - points[:, 1])
        tz[:,2] = -(points_normal[:, 2] - points[:, 2])
        for i in range(npoints):
            tx[i,:]  = np.cross(tz[i,:],np.asarray([0,1,0]))
        for i in range(npoints):
            ty[i,:]  = np.cross(tz[i,:],tx[i,:])


        Rotation_matrix[:,:,0] = tx
        Rotation_matrix[:,:,1] = ty
        Rotation_matrix[:,:,2] = tz

        return Rotation_matrix
    def From_Rotation_Matrix_To_Quaternions(self, Rotation_Matrix):
        npoints = Rotation_Matrix.shape[0]
        Quaternions = np.zeros((npoints, 4))
        for i in range(npoints):
            r = R.from_matrix(Rotation_Matrix[i,:,:])
            Quaternions[i,:] = r.as_quat()
        return Quaternions
