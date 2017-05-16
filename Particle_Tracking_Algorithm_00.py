import numpy as np
import scipy
import igraph
import matplotlib.pyplot as plt
import random
import math

#PARTICLE TRACKING

def get_pressure_field(graph, pressures, inlet_outlet_list, conductances):
    """

    :param graph: represents the pore-network model
    :param pressures: Dirichlet boundary; condition, array with all the known pressures (zero for unknown pressures)
    :param inlet_outlet_list: Neumann boundary condition; array with all the known inflows (zero for unknown or nonexistent inflows)
    :param conductances: Measure of how effectively  fluids are transported through a medium or a region. Calculated by eq. 10 (K.S.Sorbie)
    :return: The function returns the pressure field for the pore-network model considering all boundary conditions.

    """
    A = np.asmatrix(graph.laplacian(weights=conductances)).astype(np.float64)                                           #Creates the Laplacian of the system considering different conductances
    b = pressures + inlet_outlet_list                                                                                   #A*p=b, b contains the boundary conditions (inflows as well as pressures)
    A[[np.where(pressures != 0)], :] = 0                                                                                #A isn't allowed to be a singular matrix, therefore we "manipulate" the matrix to ensure that the ->
    A[np.where(pressures > 0), np.where(pressures > 0)] = 1.0                                                           #-> Dirichlet boundary conditions (known pressures) are fulfilled
    p = np.linalg.solve(A, b)                                                                                           #Solves the equation: p=A^(-1)
    return p

def get_b(graph, pressures, inlet_outlet_list,conductances):
    """

    :param graph: represents the pore-network model
    :param pressures: Dirichlet boundary; condition, array with all the known pressures (zero for unknown pressures)
    :param inlet_outlet_list: Neumann boundary condition; array with all the known inflows (zero for unknown or nonexistent inflows)
    :param conductances: Measure of how effectively  fluids are transported through a medium or a region. Calculated by eq. 10 (K.S.Sorbie)
    :return: The function returns the variable b which contains all inflows and outflows. Warning: Computing errors can occur for entries which should be zero.

    """
    A = np.asmatrix(graph.laplacian(weights=conductances)).astype(np.float64)                                           #Creates the Laplacian of the system considering different conductances
    p=get_pressure_field(graph,pressures,inlet_outlet_list,conductances)                                                #The pressure field contains the pressures in every single node in the pore-network model.
    b=np.dot(A,p)                                                                                                       #A*p=b (K.S.Sorbie)
    return b

def generate_pressure_difference_as_edge_attribute(graph,pressure_field):
    """

    :param graph: represents the pore-network model
    :param pressure_field: The pressure field contains the pressures in every single node in the pore-network model.
    :return: The function returns the pressure difference of each edge as an array. It's useful to set the pressure difference as an attribute to the edges.

    """
    edge_list = np.asarray(graph.get_edgelist())                                                                        #creates array of arrays. Includes every pair of node which are connectec with an edge.
    pressure_list = pressure_field[edge_list]                                                                           #node Id's are substituted with corresponding nodal pressures
    pressure_difference = [0] * len(pressure_list)
    for i in range(len(pressure_list)):                                                                                 #for-loop creats array with pressure differences
        pressure_difference[i] = np.abs(pressure_list[i][1] - pressure_list[i][0])
    return pressure_difference

def incident_flowlist_single_vertex(vertexID,graph,pressure_field, conductances):
    """

    :param vertexID: ID of the current node.
    :param graph: represents the pore-network model
    :param pressure_field: The pressure field contains the pressures in every single node in the pore-network model.
    :param conductances: Measure of how effectively  fluids are transported through a medium or a region. Calculated by eq. 10 (K.S.Sorbie)
    :return: The function return a list of incoming and outgoing flows to a single vertex.
            Convenction: incoming flows > 0, outgoing flows < 0

    """
    incident_nodes=np.asarray(graph.neighbors(vertexID))
    incident_edges=np.asarray(graph.incident(vertexID))
    incident_pressure_differences=pressure_field[incident_nodes]-pressure_field[vertexID]
    incident_conductances=np.asarray(conductances)[incident_edges]
    incident_flows=incident_pressure_differences*incident_conductances
    return incident_flows


#*****************************
def get_sink_or_source_flow(vertexID, graph, pressure_field, conductance):
    """

    :param vertexID: ID of the current node.
    :param graph: represents the pore-network model
    :param pressures: Boundary Condition
    :param inlet_outlet_list: Boundary Condition
    :param weights: Conductance
    :return: Function returns sum of incoming and outgoing flows. Value>0 -> sink, Value<0 -> source

    """
    return np.sum(incident_flowlist_single_vertex(vertexID, graph, pressure_field,conductance))

def get_sink_and_source_list(graph, pressures, inlet_outlet_list, pressure_field, conductance):
    """

    :param graph: Structure of the network
    :param pressures: Boundary Condition
    :param inlet_outlet_list: Boundary Condition
    :param weights: Conductance
    :return: Function returns list with all sinks and sources in the network ("Mass conservation violated")

    """
    b=pressures+inlet_outlet_list
    index_nonzero_elements=np.nonzero(b)[0]
    a=np.zeros(len(index_nonzero_elements))
    for i in range(len(index_nonzero_elements)):
        a[i]=get_sink_or_source_flow(index_nonzero_elements[i], graph, pressure_field, conductance)
    c=zip(index_nonzero_elements,a)
    return c

def get_sink_list(graph, pressures, inlet_outlet_list, pressure_field, conductance):
    """

    :param graph: Structure of the network
    :param pressures: Boundary Condition
    :param inlet_outlet_list: Boundary Condition
    :param weights: Conductance
    :return: Function returns list of sinks.

    """
    c=get_sink_and_source_list(graph, pressures, inlet_outlet_list, pressure_field, conductance)
    index_nonzero_elements,a=np.array(zip(*c))
    index=np.where(a>0)
    index_nonzero_elements=index_nonzero_elements[index]
    return index_nonzero_elements

def get_source_list(graph, pressures, inlet_outlet_list, pressure_field, conductance):
    """

    :param graph: Structure of the network
    :param pressures: Boundary Condition
    :param inlet_outlet_list: Boundary Condition
    :param weights: Conductance
    :return: Function return list of sources.

    """
    c=get_sink_and_source_list(graph, pressures, inlet_outlet_list, pressure_field, conductance)
    index_nonzero_elements,a=np.array(zip(*c))
    index=np.where(a<0)
    index_nonzero_elements=index_nonzero_elements[index]
    return index_nonzero_elements
#*****************************


def f_positive(Peclet_number):
    """

    :param Peclet_number: Single Peclet number of a tube.
    :return: Function return fraction of particles emerging in the positive directions with respect to flow velocity.
             WARNING: Peclet_number=0  is not defined, therefore the fraction f_p is set to 0.5 (other function)

    """

    if math.log(Peclet_number, 10)<=-2:
        f_p=0.5
    elif math.log(Peclet_number, 10)<=0:
        f_p=0.971+0.5065*math.log(Peclet_number, 10) + 0.1355 * math.pow(math.log(Peclet_number, 10), 2)
    elif math.log(Peclet_number, 10)<=1:
        f_p=0.971+0.051*math.log(Peclet_number, 10) - 0.022 * math.pow(math.log(Peclet_number, 10), 2)
    elif math.log(Peclet_number, 10)>1:
        f_p=1
    return f_p

def f_negative(Peclet_number):
    """

    :param Peclet_number: Single Peclet number of a tube
    :return: Function return fraction of particles emerging in the negative directions with respect to flow velocity.
             WARNING: Peclet_number=0  is not defined, therefore the fraction f_p is set to 0.5 (other function)

    """
    if math.log(Peclet_number, 10)<=-2:
        f_n=0.5
    elif math.log(Peclet_number, 10)<=0:                                                                                #log(Pe) is assumed to be smaller than 0, therefore no further elif-statements
        f_n= -(0.971 + 0.5065 * math.log(Peclet_number, 10) + 0.1355 * math.pow(math.log(Peclet_number, 10), 2)) + 1
    return f_n


def select_next_tube_and_node(vertexid, graph, pressure_field, conductances, Peclet_numbers, diffusion_coefficient,
                              beta):
    """

    :param vertexID: ID of the current node.
    :param graph: represents the pore-network model
    :param pressure_field: The pressure field contains the pressures in every single node in the pore-network model.
    :param conductances: Measure of how effectively  fluids are transported through a medium or a region. Calculated by eq. 10 (K.S.Sorbie)
    :param Peclet_number: Single Peclet number of a tube
    :param diffusion_coefficient: D_0, depending on the Material
    :param beta: Can be equal to L or (Pi*R^2)L for the cases independent of and proportional to cross-sectional area
    :return: The function returns the next step in the path of the tracking particle (edge Id, node Id)

    """
    incident_edges = np.asarray(graph.incident(vertexid))
    incident_nodes = np.asarray(graph.neighbors(vertexid))
    incident_node_edge_tuples = zip(incident_edges, incident_nodes)

    incident_flows = incident_flowlist_single_vertex(vertexid, graph, pressure_field, conductances)
    incident_length = np.array(beta)[incident_edges]
    incident_peclet_numbers = np.array(Peclet_numbers)[incident_edges]
    apparent_flows = np.zeros(len(incident_edges))

    inflow = np.asarray((incident_flows > 0)).astype(int)
    inflow_index = np.flatnonzero(inflow)
    for i in range(len(inflow_index)):
        apparent_flows[inflow_index[i]] = np.abs(f_negative(incident_peclet_numbers[inflow_index[i]]) * incident_length[
            inflow_index[i]] * diffusion_coefficient)

    outflow = np.asarray((incident_flows < 0)).astype(int)
    outflow_index = np.flatnonzero(outflow)
    for i in range(len(outflow_index)):
        apparent_flows[outflow_index[i]] = f_positive(incident_peclet_numbers[outflow_index[i]]) * (
        np.abs(incident_flows[outflow_index[i]]) + incident_length[outflow_index[i]] * diffusion_coefficient)

    cum_distribution = np.cumsum(apparent_flows) / np.sum(apparent_flows)
    random_number = random.uniform(0, 1)
    index = np.argmax(random_number < cum_distribution)
    return incident_node_edge_tuples[index]


def path_of_tracked_particle(starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta,outlet_node):
    """

    :param starting_point: Point of inlet of the tracked particle
    :param graph: Structure of the network
    :param pressures: Boundary Condition
    :param inlet_outlet_list: Boundary Condition
    :param weights: Conductance
    :param peclet_numbers: List with Peclet numbers of every single tube
    :param diffusion_coefficient: Depends on the material of the tracking particles
    :param beta: Can be set equal to the length
    :return: Function returns the path of the tracked particle through the network.

    """
    path_of_the_particle=[]
    path_of_the_particle.append(select_next_tube_and_node(starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta))


    while path_of_the_particle[-1][1] not in outlet_node:
        path_of_the_particle.append(select_next_tube_and_node(path_of_the_particle[-1][1], graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta))

    return path_of_the_particle


#TIME

def get_histogram_time(number_of_repeats, bins, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node):
    """

    :param number_of_repeats: Number of tracked particles.
    :param bins: Number of beams in the histogram.
    :param starting_point: Point of inlet of the tracked particle
    :param graph: Structure of the network
    :param pressures: Boundary Condition
    :param inlet_outlet_list: Boundary Condition
    :param weights: Conductanfe
    :param peclet_numbers: List with Peclet numbers of every single tube
    :param diffusion_constant: Depends on the material of the tracking particles
    :param beta: Can be set equal to the length
    :param average_velocity: Average velocity in each tube
    :param aspect_ratio: Radius diveded by the length of each tube
    :return: Function returns histogram for the time need to cross the network.

    """
    array_of_times = get_statistical_data_time(number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node)
    plt.hist(array_of_times, bins)
    return plt.show()

def get_statistical_data_time(number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node):
    """

    :param number_of_repeats: Number of tracked particles.
    :param starting_point: Point of inlet of the tracked particle
    :param graph: Structure of the network
    :param pressures: Boundary Condition
    :param inlet_outlet_list: Boundary Condition
    :param weights: Conductanfe
    :param peclet_numbers: List with Peclet numbers of every single tube
    :param diffusion_constant: Depends on the material of the tracking particles
    :param beta: Can be set equal to the length
    :param average_velocity: Average velocity in each tube
    :param aspect_ratio: Radius diveded by the length of each tube
    :return: Function returns an array of time needed to cross the network.

    """
    statistical_data=np.zeros(number_of_repeats)
    for i in range(number_of_repeats):
        statistical_data[i]=time_in_network(starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node)
    return statistical_data

def time_in_network(starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node):
    """

    :param starting_point: Point of inlet of the tracked particle
    :param graph: Structure of the network
    :param pressures: Boundary Condition
    :param inlet_outlet_list: Boundary Condition
    :param weights: Conductanfe
    :param peclet_numbers: List with Peclet numbers of every single tube
    :param diffusion_constant: Depends on the material of the tracking particles
    :param beta: Can be set equal to the length
    :param average_velocity: Average velocity in each tube
    :param aspect_ratio: Radius diveded by the length of each tube
    :return: Function returns time needed for a particle to cross the network.

    """
    path = path_of_tracked_particle(starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta,outlet_node)
    edge_ID, vertices=zip(*path)
    time_list=np.zeros(len(edge_ID))
    for i in range(len(edge_ID)):
        if peclet_numbers[edge_ID[i]]<=0.01:
           time_list[i]=type_A0_distribution(beta[edge_ID[i]], diffusion_constant)
        elif peclet_numbers[edge_ID[i]]<=2:
            time_list[i]=type_A1_distribution(peclet_numbers[edge_ID[i]], beta[edge_ID[i]], average_velocity[edge_ID[i]])
        elif peclet_numbers[edge_ID[i]]<0.75*math.pow(aspect_ratio[edge_ID[i]], -2):
            time_list[i]=type_B_distribution(peclet_numbers[edge_ID[i]], beta[edge_ID[i]], average_velocity[edge_ID[i]])
        else:
            time_list[i] = type_C_distribution(peclet_numbers[edge_ID[i]], beta[edge_ID[i]], average_velocity[edge_ID[i]])
    time_sum=np.sum(time_list)
    return time_sum

def get_t(peclet_number, length, average_velocity):
    """

    :param peclet_number: Single Peclet number of a tube
    :param length: Length of a tube.
    :param average_velocity: Average velocity in each tube
    :return: Function returns the mean transit time.

    """
    t= (get_t_prime(peclet_number) * length) / average_velocity
    return t

def get_sigma(peclet_number, length, average_velocity):
    """

    :param peclet_number: Single Peclet number of a tube
    :param length: Length of a tube.
    :param average_velocity: Average velocity in each tube
    :return: Function returns the standard deviation

    """
    sigma= (get_sigma_prime(peclet_number) * length) / average_velocity
    return sigma

def type_A0_distribution(length, diffusion_constant):
    """

    :param length: Length of a tube.
    :param diffusion_constant: Depends on the material of the tracking particles
    :return: Function returns random time needed to cross a single tube for tube peclet numbers  < =0.01
    """

    t = (0.63*math.pow(length, 2) )/ diffusion_constant
    sigma = (0.51*math.pow(length, 2)) / diffusion_constant
    n_1 = (3 * math.pow(sigma, 2)) / (t * (3 * math.pow(sigma, 2) + math.pow(t, 2)))
    n_2 = (math.pow(t, 3)) / (3 * math.pow(sigma, 2) * (3 * math.pow(sigma, 2) + math.pow(t, 2)))
    T = (3 * math.pow(sigma, 2) + math.pow(t, 2)) / (t)
    A_1 = n_1 * t
    A_2 = n_2 * (T - t)
    Areas = [A_1, A_2]
    cum_distribution = np.cumsum(Areas) / np.sum(Areas)
    random_number = random.uniform(0, 1)
    index = np.argmax(random_number < cum_distribution)
    time = (random.uniform(0, t), random.uniform(t, T))
    return time[index]

def type_A1_distribution(peclet_number, length, average_velocity):
    """

    :param peclet_number: Single Peclet number of a tube
    :param length: Length of a tube.
    :param average_velocity: Average velocity in each tube
    :return: Function returns random time needed to cross a single tube for 0.01 < tube peclet numbers  <=2

    """
    t= (get_t_prime(peclet_number) * length) / average_velocity
    sigma=get_sigma(peclet_number, length, average_velocity)
    n_1=(3*math.pow(sigma,2))/(t*(3*math.pow(sigma,2)+math.pow(t,2)))
    n_2=(math.pow(t,3))/(3*math.pow(sigma,2)*(3*math.pow(sigma,2)+math.pow(t,2)))
    T=(3*math.pow(sigma,2)+math.pow(t,2))/(t)
    A_1=n_1*t
    A_2=n_2*(T-t)
    Areas=[A_1,A_2]
    cum_distribution = np.cumsum(Areas) / np.sum(Areas)
    random_number = random.uniform(0, 1)
    index = np.argmax(random_number < cum_distribution)
    time=(random.uniform(0,t),random.uniform(t,T))
    return time[index]

def type_B_distribution(peclet_number, length, average_velocity):
    """

    :param peclet_number: Single Peclet number of a tube
    :param length: Length of a tube.
    :param average_velocity: Average velocity in each tube
    :return: Function returns random time needed to cross a single tube for 2<tube peclet numbers<0.75(L/R) to the 2

    """
    t=get_t(peclet_number, length, average_velocity)
    sigma=get_sigma(peclet_number, length, average_velocity)
    T_minus = t - math.sqrt(3) * sigma
    T_plus=t+math.sqrt(3)*sigma
    time=random.uniform(T_minus,T_plus)
    return time

def type_C_distribution(Peclet_Number,Length,average_velocity):
    """

    :param peclet_number: Single Peclet number of a tube
    :param length: Length of a tube.
    :param average_velocity: Average velocity in each tube
    :return: Function returns random time needed to cross a single tue for tube peclet numbers>0.75(L/R) to the 2

    """
    t=get_t(Peclet_Number,Length,average_velocity)
    sigma=get_sigma(Peclet_Number,Length,average_velocity)
    q=6*math.pow((sigma/t),2)+1
    n_1=1/(t*(2*q-1)*(q-1))
    n_2=(4*(q-1))/(t*(2*q-1))
    T_minus=t*0.5
    T_plus=q*t
    A_1=n_1*(t-T_minus)
    A_2=n_2*(T_plus-t)
    Areas=[A_1,A_2]
    cum_distribution = np.cumsum(Areas) / np.sum(Areas)
    random_number = random.uniform(0, 1)
    index = np.argmax(random_number < cum_distribution)
    time=(random.uniform(T_minus,t),random.uniform(t,T_plus))
    return time[index]

def get_t_prime(peclet_number):
    """

    :param peclet_number: Single Peclet number of a tube
    :return: Function returns the normalised transit time.

    """

    if math.log(peclet_number, 10)<=-2:
        t_prime=0.037*math.log(peclet_number, 10) + 0.111
    elif math.log(peclet_number, 10)<=0:
        t_prime=1.022+0.79572222*math.log(peclet_number, 10) - 0.284388 * math.pow(math.log(peclet_number, 10), 2) - 0.428222 * math.pow(math.log(peclet_number, 10), 3) - 0.1051111 * math.pow(math.log(peclet_number, 10), 4)
    elif math.log(peclet_number, 10)<=1:
        t_prime=1.022+0.022*math.log(peclet_number, 10) - 0.044 * math.pow(math.log(peclet_number, 10), 2)
    else: t_prime=1

    return t_prime

def get_sigma_prime(peclet_number):
    """

    :param peclet_number: Single Peclet number of a tube
    :return: Function returns the normalised standard deviation of the transit time.

    """
    if math.log(peclet_number, 10)<=-2:
        sigma_prime = 0.037 * math.log(peclet_number, 10) + 0.111
    elif math.log(peclet_number, 10)<np.negative(0.07079):
        sigma_prime=0.75185502+0.7462825*math.log(peclet_number, 10) + 0.19442751 * math.pow(math.log(peclet_number, 10), 2)
    elif math.log(peclet_number, 10)>np.negative(0.07079):
        sigma_prime=0.545+ (-4 + math.log(peclet_number, 10)) * (-0.0380761 + 0.125834 * (0.07079 + math.log(peclet_number, 10)))
    return sigma_prime

# Method (ii)

def get_t_mean_entire_network(number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node):
    list_of_times=get_statistical_data_time(number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node)
    print np.mean(list_of_times)
    return np.mean(list_of_times)

def get_experimental_standard_deviation(number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node):
    list_of_times=get_statistical_data_time(number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node)
    print np.std(list_of_times)
    return np.std(list_of_times)

def get_dispersion_coefficient_eq_15(length_total, U, number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio, outlet_node):
    D= ((length_total * U) / (2)) * np.power((get_experimental_standard_deviation(number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node) / get_t_mean_entire_network(number_of_repeats, starting_point, graph, pressure_field, conductance, peclet_numbers, diffusion_constant, beta, average_velocity, aspect_ratio,outlet_node)), 2)
    return D


















