import numpy as np 
import json

def double_gyre(x, y, t):
    A, EPS, OMEGA, pi =  0.1 , 0.25, np.pi/5, np.pi
    u = -A * pi * np.sin(pi*(EPS*np.sin(OMEGA*t)*x*x + (1 - 2 * EPS*np.sin(OMEGA*t))*x))*np.cos(pi*y)
    v = A * pi*(2 * EPS*np.sin(OMEGA*t)*x - 2 * EPS*np.sin(OMEGA*t) + 1)* np.cos(pi*(EPS*np.sin(OMEGA*t)*x*x + (1 - 2 * EPS*np.sin(OMEGA*t))*x))*np.sin(pi*y)
    V = np.stack([u,v], axis=0)
    return V
#

def four_centers(x, y, t):
    al_t = scale = 1    
    u = np.exp(-y * y - x * x)*(al_t*y*np.exp(y*y + x * x) - 6 * scale*np.cos(al_t*t)*np.sin(al_t*t)*y*y*y + (12 * scale*(np.cos(al_t*t)*np.cos(al_t*t)) - 6 * scale)*x*y*y + (6 * scale*np.cos(al_t*t)*np.sin(al_t*t)*x*x + 6 * scale*np.cos(al_t*t)*np.sin(al_t*t))*y + (3 * scale - 6 * scale*(np.cos(al_t*t)*np.cos(al_t*t)))*x)
    v = -np.exp(-y * y - x * x)*(al_t*x*np.exp(y*y + x * x) - 6 * scale*np.cos(al_t*t)*np.sin(al_t*t)*x*y*y + ((12 * scale*(np.cos(al_t*t)*np.cos(al_t*t)) - 6 * scale)*x*x - 6 * scale*(np.cos(al_t*t)*np.cos(al_t*t)) + 3 * scale)*y + 6 * scale*np.cos(al_t*t)*np.sin(al_t*t)*x*x*x - 6 * scale*np.cos(al_t*t)*np.sin(al_t*t)*x)
    V = np.stack([u,v], axis=0)
    return V
#

if __name__=='__main__':
    t_coords = np.linspace(0,10,500)
    x_coords = np.linspace(0,2,400)
    y_coords = np.linspace(0,1,200)
    spacetime_coords = np.stack(np.meshgrid(t_coords,x_coords,y_coords,indexing='ij'),axis=0)
    field = double_gyre(spacetime_coords[1],spacetime_coords[2],spacetime_coords[0])

    t_ext = [0.0,10.0]
    x_ext = [0.0,2.0]
    y_ext = [0.0,1.0]

    metadata = dict()
    metadata['dataset_name'] = 'double_gyre'
    metadata['res'] = [field.shape[1],field.shape[2],field.shape[3]]
    metadata['ext'] = [t_ext,x_ext,y_ext]
    metadata['toroidal'] = False

    json.dump(metadata,open('metadata.json','w'))
    np.save('field',field)
#
