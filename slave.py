from mpi4py import MPI
import numpy as np



"""
Add function takes list of products(strings) and concatenates them based on the machine id produced that product.
Products are merged sorted order on machine id and merged product is returned.
it takes the input in the form => [("machine id1", "product1"), ("machine id3", "product3"), ("machine id2", "product2"), ...]
"""
def add(products): 
    # sorting the list based on the machine ids
    sorted_list = sorted(products, key=lambda x: x[0]) # => [("machine id1", "product1"), ("machine id2", "product2"), ("machine id3", "product3"), ...] 
    # defining an empty string
    merged_product = ""
    # merging sorted products
    for product in sorted_list:
        merged_product += product[1]
    # returning the merged products
    return merged_product # => "product1product2product3"


"""
enhance function takes a product(string) and duplicates its first and last characters
"ABC" => "AABCC"
"""
def enhance(product):
    length = len(product)
    #returning the enhanced product
    return product[0] + product + product[length - 1]


"""
reverse function takes a product(string) and reverses it using array slicing
"ABC" => "CBA"
"""
def reverse(product):
    # reversing the product
    reversed_product = product[::-1]
    # and returning the reversed version
    return reversed_product


"""
chop function takes a product(string) chops the last character of the string. 
if product consists of only one character it returns the product without any operation.
"ABC" => "AB"
"A" => "A"
"""
def chop(product):
    length = len(product)
    if length > 1: 
        # returning the choped product
        return product[:length - 1] 
    # product consists of one character, returning it as it is
    return product 

"""
trim function takes a product(string) and deletes first and last characters of the product
if the length of the product is less than or equal to 2 it does nothing
"ABC" => "B"
"AB" => "AB"
"""
def trim(product):
    length = len(product)
    if length > 2:
        # returning the choped product
        return product[1:length - 1] 
    # product consists of less than 3 characters, returning it as it is
    return product 

"""
split function takes a product(string) and deletes the right part of it.
if number of characters is odd than middle character is counted as in the left part and returned with the left part.
"ABCD" => "AB"
"ABC" => "AB"
"""
def split(product):
    length = len(product)
    # if it is odd, 1 is added to the length to have middle character in the output
    if length % 2 != 0:
        length += 1
    # returning the splitted product
    return product[:length//2]
"""
Operate function is a helper function to prevent code repetition while handling differing operations with cycle and id.
It returns (machine id, output after operation), accumulated wear on machine after operation, and last wear factor that is added to accumulation
"""
def operate(machine_id, even_id_operations, odd_id_operations, initial_operation, cycle, initial_product, wear_factors, accumulated_wear):

    last_wf = 0
    if (machine_id) % 2 == 0: # if id is even
        idx = even_id_operations.index(initial_operation[str(machine_id)]) # we take the init operation with corresponding id,
        idx += cycle                                                       # and tracking its queue with cycle.
        idx = idx % 3
        output = "even "
        if even_id_operations[idx] == "enhance":
            accumulated_wear += wear_factors["enhance"]
            last_wf = wear_factors["enhance"]
            output = enhance(initial_product) 
        elif even_id_operations[idx] == "split":
            accumulated_wear += wear_factors["split"]
            last_wf = wear_factors["split"]
            output = split(initial_product) 
        elif even_id_operations[idx] == "chop":
            accumulated_wear += wear_factors["chop"] 
            last_wf = wear_factors["chop"]
            output = chop(initial_product)

        message = (machine_id, output)
        return message, accumulated_wear, last_wf
    else: # if id is odd we do the equivalent here.
        idx = odd_id_operations.index(initial_operation[str(machine_id)])
        idx += cycle 
        idx = idx % 2
        output = "odd "
        if odd_id_operations[idx] == "reverse":
            accumulated_wear += wear_factors["reverse"]
            last_wf = wear_factors["reverse"]
            output = reverse(initial_product)
        elif odd_id_operations[idx] == "trim":
            accumulated_wear += wear_factors["trim"] 
            last_wf = wear_factors["trim"]
            output = trim(initial_product)
        
        message = (machine_id, output) 
        return message, accumulated_wear, last_wf 

"""
In the main function communication between slave processes and master process is established and process executions are implemented based on their ranks.
"""
def main():
    # here we get the master info to communicate with the control unit room.
    master = MPI.Comm.Get_parent()
    # Here we create a communicator object to use intra-slave communications.
    intracomm = MPI.COMM_WORLD
    # Here we get the rank of the slave machine.
    rank = intracomm.Get_rank() 
    # Here we receive the broadcasted data structures by the master, so we can make use of them in later.
    odd_id_operations = master.bcast(None, root=0)
    even_id_operations = master.bcast(None, root=0)
    number_of_production_cycles = master.bcast(None, root=0)
    wear_factors = master.bcast(None, root=0)
    maintenance_threshold = master.bcast(None, root=0)
    children_information = master.bcast(None, root=0)
    parent_information = master.bcast(None, root=0) 
    initial_operation = master.bcast(None, root=0)
    leaf_products = master.bcast(None, root=0) 
    # Here we set the accumulated wear to 0
    accumulated_wear = 0 
    # This is our main loop for the operations, iterates for each cycle.
    for cycle in range(number_of_production_cycles):
        if rank == 0: # this process is not used for convenience (both master and a slave process has rank 0 )
            return 
        elif rank == 1: # root slave machine, we receive inputs from children and send them to master.
            children = children_information[str(rank)]

            child_outputs = []

            for child_id in children:
                # blocking receive for every child, since every child sends an input in every cycle.
                child_output = intracomm.recv(source=int(child_id)) 
                child_outputs.append(child_output) 

            
            input = add(child_outputs)
            message = (input)
            # Blocking send to master (dest=0) for every cycle, since we reach a final product in every cycle. 
            master.send(message, dest=0)  


        elif (str(rank) in leaf_products): # Here we handle leaf machines, they take their input from the list we defined, instead of children
            initial_product = leaf_products[str(rank)]
            parent_machine = parent_information[str(rank)] 

            message, accumulated_wear, last_wf = operate(rank, even_id_operations, odd_id_operations, initial_operation, cycle, initial_product, wear_factors, accumulated_wear) 
            
            if accumulated_wear >= maintenance_threshold: # if accumulated wear is bigger than the threshold, we send the information to master to calculate cost using cost() func.
                cost_message = np.array((rank, accumulated_wear, last_wf, cycle), dtype='i')
                req = master.Isend([cost_message, MPI.INT], dest=0) 
                req.Wait()  
                accumulated_wear = 0 
            
            # we send the product to the parent.
            intracomm.send(message, dest=int(parent_machine))  

        else: # here we handle middle machines, they take info from lower machines and send to their parents.
            children = children_information[str(rank)]
            parent_machine = parent_information[str(rank)] 

            child_outputs = []

            for child_id in children:
                child_output = intracomm.recv(source=int(child_id)) 
                child_outputs.append(child_output) 

            
            input = add(child_outputs) 

            message, accumulated_wear, last_wf = operate(rank, even_id_operations, odd_id_operations, initial_operation, cycle, input, wear_factors, accumulated_wear)   
            
            if accumulated_wear >= maintenance_threshold: # if accumulated wear is bigger than the threshold, we send the information to master to calculate cost using cost() func.
                cost_message = np.array((rank, accumulated_wear, last_wf, cycle), dtype='i')
                req = master.Isend([cost_message, MPI.INT], dest=0) 
                req.Wait()  
                accumulated_wear = 0 
            
            intracomm.send(message, dest=int(parent_machine)) 

    
    master.Disconnect()
    
if __name__ == "__main__":
    main()