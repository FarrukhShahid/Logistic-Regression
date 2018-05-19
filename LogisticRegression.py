#---------------------------------------------#
#-------| Written By: Farrukh Shahid |-------#
#---------------------------------------------#

# A Logistic Regression algorithm with regularized weights...

import numpy as np
import scipy.stats as stats

#Note: Here the bias term is considered as the last added feature 

class LogisticRegression:
    ''' Implements the LogisticRegression For Classification... '''
    def __init__(self, lembda=0.001, alpha=0.001, maxniter=2000):        
        """
            lembda= Regularization parameter...            
        """
        self.alpha = alpha
        self.maxniter = maxniter
		    self.lembda = lembda
        
        pass
	def gradient_descent(self, X, Y, cost_function, derivative_cost_function):
        '''
            Finds the minimum of given cost function using gradient descent.
            
            Input:
            ------
                X: can be either a single n X d-dimensional vector 
                    or n X d dimensional matrix of inputs            
                
                Y: Must be n X 1-dimensional label vector
                cost_function: a function to be minimized, must return a scalar value
                derivative_cost_function: derivative of cost function w.r.t. paramaters, 
                                           must return partial derivatives w.r.t all d parameters                                                           
                        
            Returns:
            ------
                thetas: a d X 1-dimensional vector of cost function parameters 
                        where minimum point occurs (or location of minimum).
        '''

        # Remember you must plot the cost function after set of iterations to
        # check whether your gradient descent code is working fine or not...
        
        
        eps=0.00001
        np.random.seed(seed=99)
        nexamples=float(X.shape[0])
        Theetas=np.random.uniform(-3,3,X.shape[1])# *
        Y=Y.astype(int)# *
        Y=Y.flatten()# *
        prevTheetas = 0
        
        i=0
        while i < (self.maxniter) and np.linalg.norm(prevTheetas - Theetas) > eps:
            td = derivative_cost_function(X, Y, Theetas)
            prevTheetas = Theetas
            Theetas = Theetas - self.alpha * td
            i += 1
        return Theetas
    def sigmoid(self,z):
        """
            Compute the sigmoid function 
            Input:
                z can be a scalar or a matrix
            Returns:
                sigmoid of the input variable z
        """

        z=np.array(-1*z,dtype=np.float32)
        print z.shape
        return np.array(1/(1.0+np.exp(z)))
    
    
    def hypothesis(self, X,theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        
        h = np.dot(X,theta)
        return h

    
    def cost_function(self, X,Y, theta):
        '''
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
                
            Return:
                Returns the cost of hypothesis with input parameters 
        '''
    
    
#         m = X.shape[0] #number of training examples
#         theta = reshape(theta,(len(theta),1))

        #y = reshape(y,(len(y),1))


        X = np.array(X)
        nexamples=float(X.shape[0])
        h = self.hypothesis(X,theta)
        a=np.log(self.sigmoid(h))
        ainv=np.log(1-self.sigmoid(h))
        part1 = -1*Y*a
        part2 = (1-Y)*ainv
        part1 = np.sum(part1-part2)/nexamples
        part2 = (self.lembda/2)*np.sum(theta**2)
        part1 = part1 + part2
        #print "cost func",part1.shape
        return part1
        


    def derivative_cost_function(self,X,Y,theta):
        '''
            Computes the derivates of Cost function w.r.t input parameters (thetas)  
            for given input and labels.

            Input:
            ------
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
            Returns:
            ------
                partial_thetas: a d X 1-dimensional vector of partial derivatives of cost function w.r.t parameters..
        '''
        X = np.array(X)
        Y = np.array(Y).flatten()
        theta = np.array(theta).flatten()
        nexamples = float(X.shape[0])
        h = self.sigmoid(self.hypothesis(X,theta))
        part1 = np.sum((h*Y-Y)*X.T+(h-h*Y)*X.T,axis=1)
        part2 = self.lembda * theta
        part1 = (part1/nexamples)+part2
        print "part 1 :",part1.shape
        return part1

    def train(self, X, Y, optimizer):
        ''' Train classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            optimizer: an object of Optimizer class, used to find
                       find best set of parameters by calling its
                       gradient descent method...
            Returns:
            -----------
            Nothing
            '''
        
#         X= np.array(X)
        self.theta = self.gradient_descent(X, Y, self.cost_function, self.derivative_cost_function)
    
    def predict(self, X):
        
        """
        Test the trained perceptron classifier result on the given examples X
        
                   
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
           
            Returns:
            -----------
                pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """
        
        num_test = X.shape[0]
        
        
#         print len(self.theta)
        res = self.sigmoid(self.hypothesis(X,self.theta))
#         print res
        return (res>0.5).astype(int)
