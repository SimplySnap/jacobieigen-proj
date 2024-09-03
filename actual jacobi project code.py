import numpy as np
import math

def find_pivot(matrix,n):
    '''This function takes in the matrix and finds the largest
    off-diagonals AKA the 'pivots'.'''
    pivot_i = pivot_j = 0 #the max elements AKA pivots' i and j values
    max_elem = 0    
    for i in range(n):
        for j in range(i+1, n):
            if abs(matrix[i, j]) >= max_elem:
                max_elem = abs(matrix[i,j])
                pivot_i = i
                pivot_j = j
    return pivot_i, pivot_j #return the max-element value, and its position

def rotate(matrix, n, piv_i, piv_j, theta, iter_num):
    ''''The function that rotates the matrix. Takes in as input the matrix, its size, 
    selected positions, angle and number of iterations we have to go through.
    First, finds the pivot, and creates the givens matrix corresponding to 
    their positions. Then, we carry out the relevant matrix multiplication.'''
    
    for count in range(0, iter_num): #Main loop
        piv_i, piv_j = find_pivot(matrix,n)

        #The below gets our value for theta
        tanval = 2*matrix[piv_i,piv_j] / (matrix[piv_i,piv_i] - matrix[piv_j,piv_j])
        theta = math.atan(tanval)/2 

        #Now, we need to get our cos and sin values
        cosval = math.cos(theta)
        sinval = math.sin(theta)

        #The below section initialises the givens matrix, and carries out the matrix mult.
        givens_matrix = np.eye(n) #Creates matrix with zeros on diagonal
        givens_matrix[piv_i, piv_i] = givens_matrix[piv_j, piv_j] = cosval
        givens_matrix[piv_i, piv_j] = -sinval
        givens_matrix[piv_j, piv_i] = sinval
        matrix = np.matmul(givens_matrix.transpose(), np.matmul(matrix, givens_matrix))
    return (matrix)

def rotate_wrapper(matrix, iter_num):
    '''Wrapper function for the rotate function. Slims down variables'''
    n = len(matrix[0])
    if isinstance(matrix, np.ndarray) == False:
        matrix = np.asarray(matrix)
    return rotate(matrix, n, 0, 0, 0, iter_num)

'''__________________________________________________________________________________
To carry out the algorithm, put in any square, symmetric matrix in the 'yourmatrix' section,
set the number of iterations you want to carry out, and watch it work!
'''
#test
yourmatrix = np.array([ \
    [ 4.0, 8, 99, 2 ], \
    [ 8, 1.0, 3.3, 1.1 ], \
    [ 99, 3.3, 3.0, 0.0 ], \
    [ 2, 1.1, 0.0, 2.0 ] ] )
final_matrix = rotate_wrapper(yourmatrix, 200)
print(final_matrix)
'''
test_2 = np.array([ \
    [ 4.0, 0.0, 0.0, 0.0 ], \
    [ 0.0, 1.0, 0.0, 0.0 ], \
    [ 0.0, 0.0, 3.0, 0.0 ], \
    [ 0.0, 0.0, 0.0, 2.0 ] ] )
print(rotate_wrapper(test_2,20)) '''

