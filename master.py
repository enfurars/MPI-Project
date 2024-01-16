import sys
from mpi4py import MPI 
import numpy as np 



"""
The cost function,
it calculates the cost as specified in description with the formula C = (A - T + 1) * W F. Returns C (Cost).
""" 
def cost(accumulated_wear, maintenance_threshold, wear_factor):
    return (accumulated_wear - maintenance_threshold + 1) * wear_factor 

"""
The master process.
It reads the input, parses it and fills the necessary data structures for easy access. 
It also communicates with the slave processes and writes the necessary outputs to a file
"""
def main(input_file_name, output_file_name):

    # let's define necessary data structures for future use 
    odd_id_operations = ["reverse", "trim"]
    even_id_operations = ["enhance", "split", "chop"] 
    number_of_machines = 0
    number_of_production_cycles = 0
    wear_factors = {"enhance": 0,
                    "reverse": 0,
                    "chop": 0,
                    "trim": 0,
                    "split": 0 } # key: name of operation, value: wear factor
    maintenance_threshold = 0
    children_information = {} # key: id of machine, value: array of children machine ids
    parent_information = {} # key: id of machine(root excluded), value: parent machine id
    initial_operation = {} # key: id of machine(root excluded), value: name of the operation
    leaf_products = {} # key: id of leaf machines, value: initial product(string)


    # Read the input file line by line and fill the lines array
    input_file = open(input_file_name, 'r') 
    lines = [] 
    while 1:
        line = input_file.readline() 
        if not line: # end of file
            break
        lines.append(line.strip()) 


    # number of machines is given in the first line
    number_of_machines = int(lines[0])


    # number of production cycles is given in the second line
    number_of_production_cycles = int(lines[1])
    
    
    # wear factor information is given in the third line
    # let's parse the 3rd line first
    wear_factor_array = lines[2].split()
    # now we can assign values to wear_factors dictionary
    wear_factors["enhance"] = int(wear_factor_array[0])
    wear_factors["reverse"] = int(wear_factor_array[1])
    wear_factors["chop"] = int(wear_factor_array[2])
    wear_factors["trim"] = int(wear_factor_array[3])
    wear_factors["split"] = int(wear_factor_array[4])


    # maintenance threshold is given in the 4th line
    maintenance_threshold = int(lines[3])


    # To fill the parent, children and initial operation dictionaries we should iterate through <number_of_machines - 1> lines in the lines array after 4th line
    for i in range(number_of_machines - 1):

        # store each adjacency information in an array temporarily => array = [child id, parent id, initial operation]
        adjacency_line_array = lines[4 + i].split()

        # filling parent dictionary => key: array[0], value: array[1]
        parent_information[adjacency_line_array[0]] = adjacency_line_array[1]

        # filling children dicrionary. => key: array[1], value: [..array[0]..] 

        # adding the first child of the machine, so initialized a new array with one element
        if adjacency_line_array[1] not in children_information: 
            children_information[adjacency_line_array[1]] = [adjacency_line_array[0]]
        # adding more children to the machine, so new children are appended to the initialized array
        else: 
            children_information[adjacency_line_array[1]].append(adjacency_line_array[0])
        
        # filling the initial operation dictionary, => key: array[0], value: array[2]
        initial_operation[adjacency_line_array[0]] = adjacency_line_array[2] 


    # To fill the leaf products dictionary, first we should find the leaf machines then assign remaining lines to them as products. We should do it in order of machine ids.

    # let's define the index of next product in the lines array. First product starts after adjacency info is done.
    idx_next_product = 4 + (number_of_machines - 1) 

    for i in range(1, number_of_machines + 1):
        # if the machine with id i has no children, it is a leaf machine
        if str(i) not in children_information:
            # then add (key: leaf machine id, value: corresponding product) to the leaf products dictionary
            leaf_products[str(i)] = lines[idx_next_product]
            # increment the next product index
            idx_next_product += 1 
    """ 
    Let's initialize MPI communication and spawn slave processes for parallel execution using Spawn 
    to create separate MPI processes for each machine.
     'args' specifies that 'slave.py' will be executed by each spawned (slave) process.
     'maxprocs' stands for the number of slave processes to be spawned.
    """
    comm = MPI.COMM_SELF.Spawn(sys.executable,
                               args=['slave.py'],
                               maxprocs=number_of_machines + 1) 
    
    """
    Here we broadcast the data structures we defined before, so we can make use of them later.
    """
    comm.bcast(odd_id_operations, root=MPI.ROOT)
    comm.bcast(even_id_operations, root=MPI.ROOT)
    comm.bcast(number_of_production_cycles, root=MPI.ROOT)
    comm.bcast(wear_factors, root=MPI.ROOT)
    comm.bcast(maintenance_threshold, root=MPI.ROOT)
    comm.bcast(children_information, root=MPI.ROOT)
    comm.bcast(parent_information, root=MPI.ROOT)
    comm.bcast(initial_operation, root=MPI.ROOT)
    comm.bcast(leaf_products, root=MPI.ROOT)

    logs = []
    # open a file to write the output
    output_file = open(output_file_name, 'w')

    # An iteration for each cycle
    for cycle in range(number_of_production_cycles):
        while True:
            if comm.iprobe(source=1): # This condition is used to handle final_product that is sent by the root slave machine.
                final_product = comm.recv(source=1)
                output_file.write(final_product) 
                output_file.write('\n') 
                break 
            elif comm.Iprobe(source=MPI.ANY_SOURCE): # This condition is for to handle the messages that are sent by other machines.
                status = MPI.Status()
                # Probe to get the size of the incoming message
                comm.Probe(source=MPI.ANY_SOURCE, status=status)
                if status.source != 1: # make sure that incoming message was not coming from the root node
                    
                    # Allocate a buffer of the size 4 (int)
                    maintenance_info = np.empty(4, dtype='i') 
    
                    # Receive the complete message
                    comm.Recv([maintenance_info, MPI.INT], source=status.source)
                    logs.append(maintenance_info)
        
        
    # let's sort the logs based on machine id and production cycle 
    sorted_logs = sorted(logs, key=lambda x: (x[0], x[3])) 

    # print the maintenance logs to file
    count = 1 # lets eliminate the new line chracter at the end of the text file by checking how many logs we wrote
    number_of_logs = len(sorted_logs)
    for log in sorted_logs:
        output_file.write(f"{log[0]}-{cost(log[1], maintenance_threshold, log[2])}-{log[3] + 1}") 
        if count != number_of_logs:
            output_file.write('\n')
        count += 1 
    
    # close the output file
    output_file.close()
    
    comm.Disconnect() 

# initializing the main function with command line arguments
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])