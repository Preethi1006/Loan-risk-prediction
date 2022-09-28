import math as M  
    
    
class LVQ :  
        
    # Here, we will create the function which computes the winning vector  
    # by Euclidean distance  
    def winner1( self, weights, sample ) :  
            
        D_0 = 0  
        D_1 = 0  
            
        for K  in range( len( sample ) ) :  
            D_0 = D_0 + M.pow( ( sample[K] - weights[0][K] ), 2 )  
            D_1 = D_1 + M.pow( ( sample[K] - weights[1][K] ), 2 )  
                
            if D_0 > D_1 :  
                return 0  
            else :   
                return 1  
    
    # Here, we will create the function which here updates the winning vector       
    def update1( self, weights, sample, J, alpha1 ) :  
        for k in range(len(weights)) :  
            weights[J][k] = weights[J][k] + alpha1 * ( sample[k] - weights[J][k] )   
    
# Driver code  
def main() :  
    
    # Here, we are training Samples ( g, h ) with their class vector  
    P =  [[ 0, 0, 1, 1 ],  [ 1, 0, 1, 0 ],   
          [ 0, 0, 0, 1 ], [ 0, 1, 1, 0 ],  
          [ 1, 1, 1, 0 ], [ 1, 0, 1, 1 ],]   
    
    Q = [ 0, 1, 0, 1, 0, 1 ]  
    g, h = len( P ), len( P[0] )  
        
    # HEre, we will initialize the weight ( h, c )  
    weights = []  
    weights.append( P.pop( 0 ) )  
    weights.append( P.pop( 1 ) )  
    
    # Samples used in weight initialization will  
    # not use in training  
    g = g - 2  
        
    # training  
    ob1 = LVQ()  
    epochs1 = 10  
    alpha1 = 0.1 
        
    for k in range( epochs1 ) :  
        for o in range( g ) :  
                
            # Sample selection  
            T = P[o]  
                
            # Compute winner  
            J = ob1.winner1( weights, T )  
            
            # Update weights  
            ob1.update1( weights, T, J, alpha1 )  
                
    # classify new input sample  
    T = [ 0, 1, 1, 0 ]  
    J = ob1.winner1( weights, T )  
    print( "Sample T belongs to class : ", J )  
    print( "Trained weights : ", weights )  
        
if __name__ == "__main__":  
    main()  