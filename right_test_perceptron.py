
# test 


import perceptron


def test_prediction():

    x = [10,0]
    y = [-10,0]
    object1 = perceptron.Perceptron(0.5)

    assert 1 == object1.prediction(x), "output of x= 10 should be 1"
    assert 0 == object1.prediction(y), "output of x= -10 should be 0"
    print("============================================")
    print("OK!! <<test pass>>")
    print("============================================")





def test_train():

    object2 = perceptron.Perceptron(0.5)
    object2.weights[0] = 1
    object2.weights[1] = 1
    object2.bias = 1.0
    
    # 10km/s isn't over 50km/s, so y is assigned 0. 
    # Then activation = 1 becouse of z = 11>0 .

    object2.train([10,0],0)

    assert -4 == object2.weights[0],"weight[0] must be no updated by"
    assert 1 == object2.weights[1],"weight[1] must be no updated by"
    assert 0.5 == object2.bias,"There are must be no update bias "
    print("============================================")
    print("OK!! <<test pass>>")
    print("============================================")



    """
    Now object2.weights are changed  
    
    w[0] = -4 
    w[1] = 1 
    
    And bias is ..
    
    bias = 0.5

    """
    # 60km/s is over 50km/s so y is assigned 1
    # Then activation = 0 becouse of z =  -239.5<0 .
    object2.train([60,0],1)

    assert 26 == object2.weights[0],"weight[0] must be updated by"
    assert 1 == object2.weights[1],"weight[1] must be updated by"
    assert 1 == object2.bias,"bias must be updated"
    print("===========================================")
    print("OK!!  <<test pass>>")
    print("===========================================")


    """
    Now object2.weigths are changed 
    
    w[0] = 26
    w[1 = 1
    
    And bias is ..
    
    bias = 1
    
    """
    object2.train([])



if __name__ == '__main__':
        test_prediction()
        test_train()









