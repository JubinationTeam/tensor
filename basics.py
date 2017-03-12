#import dependencies
import numpy as np

#train data
train_x = np.array([[0,0,0],
                    [0,1,0],
                    [1,0,0]])
train_y = np.array([[1],
                    [0],
                    [1]])

#predict data
predict_x = np.array([[0,0,0],
                    [0,1,0],
                    [1,0,0],
                    [1,1,0],
                    [0,0,1],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1]])
predict_y = np.array([[1],
                    [0],
                    [1],
                    [0],
                    [1],
                    [0],
                    [1],
                    [0]])
error=0.00000001
nh1=25

#sizes
datasize=len(train_x)
ni=len(train_x[0])
no=len(train_y[0])



#weight initialization
np.random.seed(1)
w0=2*np.random.rand(ni,nh1+1)-1
w1=2*np.random.rand(nh1,no+1)-1



#sigmoid
def sigmoid(z,Deriv=False):
    if Deriv :
        return z*(z-1)
    return 1/(1+np.exp(z))

#feed forward
def predict(x,w0,w1):
    z0=x.dot(w0)
    a0=sigmoid(z0)
    z1=a0.dot(w1)
    a1=sigmoid(z1)
    y_=a1
    return [z0,z1,a0,a1]

#back propagation
def back_propagation(x,y,w0,w1,z0,z1,a0,a1):
    y_=a1
    error=0.5*(y-y_)*(y-y_)
    e1=y-y_
    delta1=sigmoid(a1,Deriv=True)
    del_w1=a0.T.dot(e1*delta1)
    w1+=del_w1
    e2=np.dot(e1*delta1,w1.T)
    delta2=sigmoid(a0,Deriv=True)
    del_w0=x.T.dot(e2*delta2)
    w0+=del_w0
    return np.average(error)

#trainer
def train(x,y,w0,w1,error_accepted):
    diff_past=1000000.0
    diff=0.0000001
    iteration=0
    while (diff>error_accepted) :
        val=predict(x,w0,w1)
        diff_now=back_propagation(x,y,w0,w1,val[0],val[1],val[2],val[3])
        diff=diff_past-diff_now
        diff_past=diff_now
        iteration+=1
    return iteration
    

def operate(predict_x,predict_y,train_x,train_y,w0,w1,error_accepted) :
    print("Before Training")
    predict_print(predict_x,predict_y,w0,w1) 
    print("----------------------------")
    
    print("Training.....")
    print("After "+str(train(train_x,train_y,w0,w1,error_accepted))+" iterations..")  
    print("----------------------------\n")
    
    print("After Training")
    predict_print(predict_x,predict_y,w0,w1)
    return

def predict_print(predict_x,predict_y,w0,w1):
    print("================")
    print("Prediction")
    predicted=predict(predict_x,w0,w1)[3]
    true=0
    false=0
    confidence=0.0
    total=0
    for i in range(len(predicted)) :
        if(predicted[i]>=0.5) :
            confidence+=(predicted[i]-0.5)*100/0.5
            if (predict_y[i]>0):
                true+=1
            else :
                false+=1
        else :
            confidence+=(0.5-predicted[i])*100/0.5
            if (predict_y[i]<1):
                true+=1
            else :
                false+=1
        total+=1
    confidence/=total  
    accuracy=0
    if(true+false)!=0:
        accuracy=(true*100)/(true+false)
    throughput=(accuracy*confidence)/100
    print("Accuracy of "+str(accuracy)+"%")
    print("Confidence of "+str(confidence)+"%") 
    print("Throughput of "+str(throughput)+"%") 
    print("================\n")
    return
operate(predict_x,predict_y,train_x,train_y,w0,w1,error)    