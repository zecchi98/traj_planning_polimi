import numpy as np
from geomdl import NURBS
from geomdl.fitting import interpolate_curve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from geomdl import utilities
from geomdl import BSpline
from utility_functions import Utility_functions
from geomdl.visualization import VisMPL
from matplotlib import cm
from scipy.spatial.transform import Rotation as R


if __name__ == '__main__':

    ''' Control points in mm'''
    # CONVEX SURFACE
    pnts_convex = [[0,0,0,1], [12.5,0,5,1], [25,0,0,1],
            [0,12.5,5,1], [12.5,12.5,12.5,1], [25,12.5,5,1],
            [0,25,0,1], [12.5,25,5,1], [25,25,0,1]]

    # CONCAVE SURFACE
    pnts_concave = [[0,0,0,1], [12.5,0,-5,1], [25,0,0,1],
            [0,12.5,-5,1], [12.5,12.5,-12.5,1], [25,12.5,-5,1],
            [0,25,0,1], [12.5,25,-5,1], [25,25,0,1]]

    # RULED SURFACE
    pnts_ruled = [[0,0,0,1], [15,0,0,1], [30,0,0,1],
            [0,10,5,1], [15,10,5,1], [30,10,5,1],
            [0,20,6,1], [15,20,6,1], [30,20,6,1]]
    # KNOT VECTORS
    knots = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]

    '''Build the NURBS Surface'''
    Utilities = Utility_functions()
    surf = Utilities.NURBS_Surface_building(pnts_ruled,knots)

    ''' Visualize only surface'''
    vis_config = VisMPL.VisConfig(legend=True, axes=True, figure_dpi=120)
    surf.vis = VisMPL.VisSurface(vis_config)
    surf.render(colormap=cm.summer)
    dim_u, dim_v, surface_size_u, surface_size_v, surface_length = Utilities.Surface_dimensions(surf)
    surfpts, initial_curve_eval_points, initial_curve_points, initial_path, points, points_normal, points_su, points_sv, curve_final  = Utilities.Trajectory_generation(surf,surface_length,surface_size_u, surface_size_v)
    ''' Rotation Matrix computation to have the z axis of the EE aligned with the normal'''
    Rotation_matrix = Utilities.Rotation_Matrix_from_Normal_Vector(points,points_normal,points_su,points_sv)
    ''' From Rotation Matrix to Quaternion'''
    Quaternions = Utilities.From_Rotation_Matrix_To_Quaternions(Rotation_matrix)


    file=open("output_traj.txt","w")
    dim=len(points)
    print(dim)
    for indx in range(0,dim):
            point_str=str(points[indx][0])+ " " +str(points[indx][1])+ " " +str(points[indx][2])
            quat_str=str(Quaternions[indx][0])+ " " +str(Quaternions[indx][1])+ " " +str(Quaternions[indx][2])+ " " +str(Quaternions[indx][3])
            output_string=point_str + " " + quat_str + "\n"
            
            file.write(output_string)

    file.close()


    ''' Visualize polishing path'''
    fig2 = plt.figure(figsize=(10.67, 8), dpi=96)
    ax2 = Axes3D(fig2)
    surfpts = np.array(surf.evalpts)
    ax2.plot_trisurf(surfpts[:, 0], surfpts[:, 1], surfpts[:, 2], cmap=cm.summer, alpha=0.5)
    ax2.plot(points[:, 0],points[:, 1], points[:, 2], color='red', linestyle='-', linewidth=3)
    ax2.quiver(points[0, 0], points[0, 1], points[0, 2], points_normal[0, 0], points_normal[0, 1], points_normal[0, 2], length=0.1, normalize=True,color='r')
    ax2.quiver(points[:, 0], points[:, 1], points[:, 2], points_normal[:, 0], points_normal[:, 1], points_normal[:, 2], length=0.1, normalize=True,color='k')
    ax2.quiver(points[:, 0], points[:, 1], points[:, 2], points_su[:, 0], points_su[:, 1], points_su[:, 2], length=0.1, normalize=True,color='k')
    ax2.quiver(points[:, 0], points[:, 1], points[:, 2], points_sv[:, 0], points_sv[:, 1], points_sv[:, 2], length=0.1, normalize=True,color='k')
    ax2.set_xlabel('$u$')
    ax2.set_ylabel('$v$')
    ax2.set_zlim3d(top=surface_length)
    fig2.suptitle('Polishing path')
    plt.show()

