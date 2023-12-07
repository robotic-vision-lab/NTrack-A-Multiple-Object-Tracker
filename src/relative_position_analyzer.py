import numpy as np
from filterpy import stats
from numpy import polyfit



def get_consistent_loc_linear_in_pos_byte(trackers, neighbors,  n_neighbor):


    gaussians = []
    neighbor_weight = []
    ev = 10
    for id, relative_vecs in neighbors.items():
        #if id in [11,12]: continue
        if id in trackers: # if neighbor detected in the recent  frame
            c, v = trackers[id].particle_filter.estimate()
            n_sample = len(relative_vecs)
            if n_sample>1:
                #n_sample =1
                rv = np.array(relative_vecs)  # convert ot np array for easier indexing
                neighbor_weight.append(np.sum(rv[:,0]))  # prioritize most recent neighbors with maximum number of history
                #frm, relative_x, relative_y = rv[:,0], rv[:,1], rv[:,2]
                n_point =9
                nx, ny, x, y = rv[:, 3][-n_point:], rv[:, 4][-n_point:], rv[:, 5][-n_point:], rv[:, 6][-n_point:]
                px = polyfit(nx, x, 1)
                py = polyfit(ny, y, 1)
                # vx = (np.sum((np.polyval(px, nx) - x)**2))/n_sample+ev #np.sqrt(
                # vy = (np.sum((np.polyval(py, ny) - y)**2))/n_sample+ev ##np.sqrt(
                vx = (np.sum((np.polyval(px, nx) - x) ** 2)) / n_sample + ev  # np.sqrt(
                vy = (np.sum((np.polyval(py, ny) - y) ** 2)) / n_sample + ev  ##np.sqrt(
                vx =  np.sqrt(vx)
                vy = np.sqrt(vy)
                x_poly, y_poly = np.poly1d(px), np.poly1d(py)
                gaussians.append([ [x_poly(c[0]), y_poly(c[1]) ], np.diag([vx+v[0], vy+v[1]]) ])
            else:
                neighbor_weight.append(relative_vecs[0][0])
                gaussians.append([ [c[0]+relative_vecs[0][1] , c[1]+relative_vecs[0][2] ],
                                   np.diag([100+v[0], 10+v[1]]) ])

    if len(gaussians)<1:
        return None, None
    n_neighbor = min(n_neighbor, len(neighbors))
    indx = np.argsort(neighbor_weight)[-n_neighbor:] # consider  most weighted n_neighbor
    gaussians = [gaussians[i] for i in indx]


    combined_meu, combined_cov = gaussians[0]
    combined_meu = np.array(combined_meu)
    for i in range(1, len(gaussians)): #more than 1 neighbor with good data
        combined_meu, combined_cov = stats.multivariate_multiply(combined_meu, combined_cov,
                                    gaussians[i][0],gaussians[i][1])

    return combined_meu, combined_cov