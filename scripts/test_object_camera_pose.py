import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from basic_utils.object_point_cloud_utils.geometry_utils import xyz_yaw_to_tf_matrix,xyz_yaw_pitch_to_tf_matrix,transform_points

class CameraPoseTests(unittest.TestCase):
    def test_matches_occupancy_camera_extrinsics(self):
        body_to_optical=np.array([[0,-1,0],[0,0,-1],[1,0,0]])
        for yaw in [-2.7,0.,1.1]:
            for pitch in [-np.pi/6,0.,np.pi/6]:
                camera=Rotation.from_euler('xyz',[pitch+np.pi/2,np.pi,yaw+np.pi/2]).as_matrix()
                actual=xyz_yaw_pitch_to_tf_matrix(np.array([1.,2.,.88]),yaw,pitch)
                np.testing.assert_allclose(actual[:3,:3],camera@body_to_optical,atol=1e-14)
                np.testing.assert_array_equal(actual[:3,3],[1.,2.,.88])
    def test_level_identical_to_old_transform(self):
        xyz=np.array([1.,2.,.88])
        np.testing.assert_array_equal(xyz_yaw_pitch_to_tf_matrix(xyz,.7,0.),xyz_yaw_to_tf_matrix(xyz,.7))
    def test_look_up_forward_ray(self):
        out=transform_points(xyz_yaw_pitch_to_tf_matrix(np.array([0.,0.,.88]),0.,np.pi/6),np.array([[2.,0.,0.]]))
        np.testing.assert_allclose(out,[[np.sqrt(3),0.,1.88]])

if __name__=='__main__':unittest.main()
