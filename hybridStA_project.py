#!/bin/env python

import warnings
warnings.filterwarnings("ignore")
import argparse
import starfile
import numpy as np
from eulerangles import euler2matrix
from eulerangles import matrix2euler

parser = argparse.ArgumentParser(description="""
                                 
Project subtomograms on the 2D untilted image.\n
NOTE: the script expects relion4 format input particles.star file.\n
It outputs particle coordinate  star files per tomogram as well as the global coordinate file,\n
which can be then used to extract particles in relion:\n
relion_preprocess_mpi --coord_list my_coords/autopick.star ...\n 
This script is based off the implementation from https://github.com/KudryashevLab/dyn2rel
                                 
""",
formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

general_group = parser.add_argument_group('General options')

general_group.add_argument("--apix", 
                    type=float, 
                    required=True, 
                    help="Unbinned pixel size of the data."
                    )
general_group.add_argument("--tomo_size_X", 
                    type=float, 
                    required=True, 
                    help="Unbinned tomogram size along X dimension in pixels."
                    )
general_group.add_argument("--tomo_size_Y", 
                    type=float, 
                    required=True, 
                    help="Unbinned tomogram size along Y dimension in pixels."
                    )
general_group.add_argument("--tomo_size_Z", 
                    type=float, 
                    required=True, 
                    help="Unbinned tomogram size along Z dimension in pixels/thickness."
                    )
general_group.add_argument("--input_star", 
                    type=str, 
                    required=True, 
                    help="Input particles.star file in the relion4 tomo format"
                    )
general_group.add_argument("--output_path", 
                    type=str, 
                    required=True, 
                    help="Output path where the coordinate files will be written"
                    )
general_group.add_argument("--movie_prefix", 
                    type=str, 
                    required=True, 
                    help="Relative path to the untilted images in the relion directory"
                    )
general_group.add_argument("--imod_path", 
                    type=str, 
                    required=True, 
                    help="Path to the imod-style tomogram folders that contain .tlt and .xf files"
                    )
args = parser.parse_args()

apix=args.apix
tomo_size = np.array([args.tomo_size_X,args.tomo_size_Y,args.tomo_size_Z])
input_star = args.input_star
output_path = args.output_path
movie_prefix = args.movie_prefix
imod_path = args.imod_path

def hybridStA_project(apix, tomo_size, input_star, output_path, movie_prefix, imod_path):
    df = starfile.read(input_star) #relion4 --tomo format for now
    #re-center XYZ
    df['particles']['rlnCoordinateX'] = df['particles']['rlnCoordinateX'] - (df['particles']['rlnOriginXAngst']/apix) #recenter with shifts from the input starfile
    df['particles']['rlnCoordinateY'] = df['particles']['rlnCoordinateY'] - (df['particles']['rlnOriginYAngst']/apix)
    df['particles']['rlnCoordinateZ'] = df['particles']['rlnCoordinateZ'] - (df['particles']['rlnOriginZAngst']/apix)
    
    tids = np.zeros(len(df['particles']['rlnTomoName'].to_list())) #get a list of tomogram IDs to loop over
    for i,j in enumerate(df['particles']['rlnTomoName'].to_list()):
        tids[i] = int(j.split('_')[1]) #expects rlnTomoName entries in the tomogram_%03d format
    tid = np.unique(tids)
    
    coords_star = open(output_path + 'autopick.star', 'w') #initiate the global coordinate output starfile
    coords_star.write('\ndata_coordinate_files\n\nloop_\n');cnt = 1
    coords_star.write('_rlnMicrographName\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
    coords_star.write('_rlnMicrographCoordinates\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1

    for i in tid:
        i = int(i)
    
        coords_star.write(movie_prefix + f"tomogram_{i:03d}.mrc\t" + output_path + f"tomogram_{i:03d}_autopick.star\n") #initiate the per-tomogram output coordinate starfile
    
        df_tomo = df['particles'][df['particles']['rlnTomoName'].isin(['tomogram_' + str('{:0>3}'.format(i))])]
        tlt_name = imod_path + f"tomogram_{i:03d}/tomogram_{i:03d}.tlt" #read in the corresponding tlt file
        tlt = np.loadtxt(tlt_name);
    
        if 0 in tlt:
                hd_idx = np.where(tlt==0)[0][0] #locate the index of the untilted image in the stack
        else:
            continue
        
        xf_name = imod_path + f"tomogram_{i:03d}/tomogram_{i:03d}.xf" #read in the corresponding xf file
        xf = np.loadtxt(xf_name);
        ct = np.cos(tlt[hd_idx] * np.pi / 180. )
        st = np.sin(tlt[hd_idx] * np.pi / 180. )
        Ttlt = np.array([[ct, 0, -st, 0], [0, 1, 0, 0,], [st, 0, ct, 0], [0, 0, 0, 1]]) #transformation matrix from the tlt
        Txf = np.identity(4) #transoformation matrix from the xf
        Txf[0,:] = [xf[hd_idx,0],xf[hd_idx,1], 0, xf[hd_idx,4]]
        Txf[1,:] = [xf[hd_idx,2],xf[hd_idx,3], 0, xf[hd_idx,5]]
        Txyz = np.linalg.inv(np.matmul(Ttlt,Txf)) #the total coordinates transformation matrix
        Teul = np.matmul(Ttlt,Txf) #the total angles transformation matrix
        tomo_center = tomo_size/2 + 1; stack_center = np.array([tomo_size[1],tomo_size[0]])/2 + 1;
        pos = np.array([df_tomo['rlnCoordinateX'].to_list(),df_tomo['rlnCoordinateY'].to_list(),df_tomo['rlnCoordinateZ'].to_list()]).T #get the 3D XYZ

        pos[:,0] = pos[:,0] - tomo_center[0];pos[:,1] = pos[:,1] - tomo_center[1];pos[:,2] = pos[:,2] - tomo_center[2] #center the origin 
    
        pos = np.matmul(np.insert(pos,3,1.0,axis=1),Txyz.T); #transform the coordinates
    
        pos[:,0] = pos[:,0] + stack_center[0]; #shift the origin back
        pos[:,1] = pos[:,1] + stack_center[1];
        pos = pos[:,0:3]

        df_tomo = df_tomo[np.logical_and(pos[:,0]>0,pos[:,1]>0)] #remove failed
        pos = pos[np.logical_and(pos[:,0]>0,pos[:,1]>0),:]

        df_tomo = df_tomo[np.logical_and(pos[:,0]<tomo_size[1],pos[:,1]<tomo_size[0])]
        pos = pos[np.logical_and(pos[:,0]<tomo_size[1],pos[:,1]<tomo_size[0]),:]

        df_tomo = df_tomo.reset_index(drop=True)
        with open(output_path + f"tomogram_{i:03d}_autopick.star", 'w') as f:
            f.write('\ndata_coordinates\n\nloop_\n');cnt = 1
            f.write('_rlnCoordinateX\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnCoordinateY\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnCoordinateZ\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnTomoParticleId\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnAngleRot\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnAngleTilt\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnAnglePsi\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnOriginXAngst\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnOriginYAngst\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnOriginZAngst\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            f.write('_rlnRandomSubset\t#' + str('{:>}'.format(cnt)) + '\n');cnt = cnt+1
            for k in range(len(df_tomo)):
                euler = np.array([df_tomo['rlnAngleRot'][k], df_tomo['rlnAngleTilt'][k], df_tomo['rlnAnglePsi'][k]]) #get the euler angles from StA
                R = euler2matrix(euler,axes='zyz',intrinsic=True,right_handed_rotation=True) #convert the euler angles to matrix
                R2d = np.matmul(R,Teul[0:3,0:3]) #transform the angles
                euler2d = matrix2euler(R2d,axes='zyz',intrinsic=True,right_handed_rotation=True) #convert from matrix to euler triplets
                f.write(str('{:.6f}'.format(pos[k,0])) + '\t' \
                        + str('{:.6f}'.format(pos[k,1])) + '\t' \
                        + str('{:.6f}'.format(df_tomo['rlnCoordinateZ'][k])) + '\t' \
                        + str('{:.6f}'.format(df_tomo['rlnTomoParticleId'][k])) + '\t' \
                        + str('{:.6f}'.format(euler2d[0])) + '\t' \
                        + str('{:.6f}'.format(euler2d[1])) + '\t' \
                        + str('{:.6f}'.format(euler2d[2])) + '\t' \
                        + str(0) + '\t' + str(0) + '\t' + str(0) + '\t' \
                        + str('{:.6f}'.format(df_tomo['rlnRandomSubset'][k])) + '\n')
    coords_star.close()

hybridStA_project(apix, tomo_size, input_star, output_path, movie_prefix, imod_path)
