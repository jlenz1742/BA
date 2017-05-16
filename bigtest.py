import igraph
import random
import numpy as np
import scipy as sp
import Particle_Tracking_Algorithm_Final_Version
import Particle_Tracking_Algorithm_00
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import scipy.stats as st




#Build the Graph
"""
Graph.GRG() generates a geometric random graph: n points are chosen randomly and uniformly inside the unit square
and pairs of points closer to each other than a predefined distance d are connected by an edge.
"""
g=igraph.Graph.GRG(1000, 0.075)
igraph.summary(g)

#Plotting the Graph with vertex-ID
g.vs["name"]=np.arange(g.vcount())
g.vs["label"]=g.vs["name"]
layout=g.layout_kamada_kawai()
visual_style={}
visual_style["vertex_size"]=20
visual_style["vertex_label"]=g.vs["name"]
#igraph.plot(g, **visual_style)

#Get inlet and outlet nodes
x_coord=g.vs['x']
inlet_condition=np.asarray(x_coord)<0.08
inlet_nodes=np.nonzero(inlet_condition)[0]
outlet_condition=np.asarray(x_coord)>0.92
outlet_nodes=np.nonzero(outlet_condition)[0]

#set the pressures (Dirichlet)
pressures=np.zeros(g.vcount())
pressures[outlet_nodes]=1

#set the inlets and outlets (Neumann)
Q_inflow_total=6e-12                                                                   #Total inflow on left-hand side to the system
random_inflow_probabilities=np.random.dirichlet(np.ones(len(inlet_nodes)),size=1)      #split total inflow to the single inlet nodes
random_inflow_list=Q_inflow_total*random_inflow_probabilities
inlet_outlet_list=np.zeros(g.vcount())
inlet_outlet_list[inlet_nodes]=random_inflow_list                                      #list wich contains the boundary conditions (Inflow)

#Get fluid viscosity and diffusion_coefficient D_0
fluid_viscosity=1e-3                                                                   #Both in SI-units
diffusion_constant=2.5e-7


#Geometries
g.es["Radius"]=np.random.uniform(1e-6,20e-6,g.ecount())                                 #Radius is set randomly for each edge. Min: 1e-6, Max: 20e-6
R=g.es["Radius"]
g.es["Aspect Ratio"]=np.random.uniform(0.025,0.5,g.ecount())                            #Aspect ratio lies between 0.025 and 0.5 for porous media, can be set randomly
AR=g.es["Aspect Ratio"]
g.es["Length"]=np.asarray(g.es["Radius"])/g.es["Aspect Ratio"]                          #Length is Calculated from Radius and aspect ration (AR=R/L)
L=g.es["Length"]
g.es["Conductance"]=(np.pi*np.power(g.es["Radius"],4))/(8*fluid_viscosity*np.asarray(g.es["Length"]))   #Is calulated with eq. 10 (K.S.Sorbie)
C=g.es["Conductance"]
g.es["Area"]=np.pi*np.power(g.es["Radius"],2)
A=g.es["Area"]
g.es["Beta"]=np.asarray(g.es["Area"])/g.es["Length"]
B=g.es["Beta"]

#Get the pressure field from eq. 10 (K.S.Sorbie)
pressure_field=Particle_Tracking_Algorithm_testxx_first_time_outlet_node_particle_out.get_pressure_field(g,pressures,inlet_outlet_list,C)

#Other Attributes
g.es["Pressure Difference"]=Particle_Tracking_Algorithm_testxx_first_time_outlet_node_particle_out.generate_pressure_difference_as_edge_attribute(g,pressure_field)     #Pressuredifference (absolute value)
PD=g.es["Pressure Difference"]
g.es["Average Velocity"]=(np.asarray(g.es["Conductance"])*g.es["Pressure Difference"])/(g.es["Area"])                   #Average velocity in each tube
AV=g.es["Average Velocity"]
#g.es["Peclet Number"]=(np.asarray(g.es["Average Velocity"])*g.es["Length"])/(4*diffusion_constant)                      #Peclet number in each tube
#P=g.es["Peclet Number"]

#Specifications of the network model
thickness=np.mean(g.es["Radius"])
width=20
length=100
inflow_total=1e-6
area=width*thickness
mean_fluid_velocity=inflow_total/area




#print Particle_Tracking_Algorithm_testxx_first_time_outlet_node_particle_out.get_histogram_time(10000,100,0,g,pressure_field,C,P,diffusion_constant,B,AV,AR,outlet_nodes)

print Particle_Tracking_Algorithm_testxx_first_time_outlet_node_particle_out.select_next_tube_and_node(0,g,pressure_field,C,diffusion_constant,L,R,fluid_viscosity)