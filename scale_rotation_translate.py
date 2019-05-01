import numpy as np
import numpy.linalg
from read_vertices_faces_obj import retrieve_vertices_faces

# Relevant links:
#   - http://stackoverflow.com/a/32244818/263061 (solution with scale)
#   - "Least-Squares Rigid Motion Using SVD" (no scale but easy proofs and explains how weights could be added)


# Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
# with least-squares error.
# Returns (scale factor c, rotation matrix R, translation vector t) such that
#   Q = P*cR + t
# if they align perfectly, or such that
#   SUM over point i ( | P_i*cR + t - Q_i |^2 )
# is minimised if they don't align perfectly.
def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(mesh1_V, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


# Testing

np.set_printoptions(precision=6)

[mesh1_V,mesh1_F]=retrieve_vertices_faces('/home/administrator/Desktop/kiran/50_rotate.obj')
[mesh2_V,mesh2_F]=retrieve_vertices_faces('./testing_stright.obj')
mesh1_V=np.asarray(mesh1_V)
mesh2_V=np.asarray(mesh2_V)

c, R, t = umeyama(mesh1_V,mesh2_V)


print "R =\n", R
print "c =", c
print "t =\n", t
print
print "Check:  a1*cR + t = a2  is", np.allclose(mesh1_V.dot(c*R) + t, mesh2_V)
obtained=mesh1_V.dot(c*R) + t
print obtained
err = ((mesh1_V.dot(c * R) + t - mesh2_V) ** 2).sum()
print "Residual error", err
with open('res1.obj','w') as f:
	for vrt in obtained:
		f.write('v {} {} {}\n'.format(vrt[0],vrt[1],vrt[2]))
	for face in mesh1_F:
		f.write('f {} {} {}\n'.format(face[0],face[1],face[2]))

