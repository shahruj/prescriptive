# -*- coding: utf-8 -*-
"""
@author: Shahruj Rashid
"""
from random import random
from random import uniform
import math
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import numpy as np
import pickle
import random
from matplotlib  import cm 
import time
import sys

#iterat = int(sys.argv[1])
#part = int(sys.argv[2])

iterat = 100
part =30
model2 = pickle.load(open("model.dat", "rb"))
#model2 = keras.models.load_model('my_model_2')
#model = pickle.load(open("model.dat", "rb"))


#function returns the output of the model when passed in a particle's parameters
#add models here  

def Invoke_Model(payload):
    ''' SERVICE NUMBER 4. The function  is invoke inference from the endpoint 

        payload = {data:data,enpoint_name:endpoint_name}

        sample model_data : [0.23,0.45,0.34,0.35,0.24,0.69,0.92,0.85]

        sample endpoint_name: 'sagemaker-tensorflow-serving-2021-02-21-fypmodel-endpoint'
    '''
    import boto3
    client = boto3.client('runtime.sagemaker')
    data = ast.literal_eval(payload['data'])
    response = client.invoke_endpoint(EndpointName=payload['endpoint_name'],
                                      ContentType='application/json',
                                      Body=json.dumps(data))

    result = json.loads(response['Body'].read().decode())
    return result['predictions']

def evaluate_position(temp,online,limit):
	if not online:
	    temp = np.asarray(temp)
	    CNNout=model2.predict(temp)
	    CNNout = (CNNout*(limit[1]-limit[0]))+limit[0]
	    #CNNout = (CNNout*(82.599225-2.331808))+2.331808
	    return CNNout
    else:
        SMout = Invoke_Model(temp)
	    SMout = (SMout*(limit[1]-limit[0]))+limit[0]
	    return SMout

def fitness(arr):
    ideal = 100
    temp = np.asarray(arr)
    temp = np.absolute(arr-ideal)
    return temp    

       
class Population:
    def __init__(self,n,dim,x,exclude,bound,is_online):
        self.pop=np.zeros((n,dim+2))
        self.bestpop = np.zeros((4,dim+1))
        self.dimension = dim
        self.pop_size = n
        self.x = x
        self.exclude = exclude
        self.bounds = bound[0:8]
        self.limit = bound[8]
        self.is_online = is_online
        for i in range(0,n):
            for j in range(0,self.dimension):
                self.pop[i,j] = random.randint(int(self.bounds[j][0]), int(self.bounds[j][1]))
    
    def evaluate_gene(self):
        temp=np.divide(np.subtract(self.pop[:,0:self.dimension],self.bounds[:,0]),np.subtract(self.bounds[:,1],self.bounds[:,0]))
        self.pop[:,self.dimension]=evaluate_position(temp,self.is_online,self.limit).transpose()
        
    def evaluate_fitness(self):
        self.pop[:,self.dimension+1]=fitness(self.pop[:,self.dimension])
        
    def selection(self):
        arg = self.pop[:,self.dimension+1].argsort()[0:4]
        self.bestpop = self.pop[arg,:]
    
    def crossover(self):
        self.pop[0:4,:]=self.bestpop
        for i in range(4,int(4+(self.pop_size-4)*0.5)):
            sel = [0,1,2,3]
            a = sel.pop(random.randint(0,len(sel)-1))
            b = sel.pop(random.randint(0,len(sel)-1))
            self.pop[i,0:self.dimension//2] = self.bestpop[a,0:self.dimension//2]
            self.pop[i,self.dimension//2:self.dimension] = self.bestpop[b,self.dimension//2:self.dimension]
               
    def mutate(self):
        for i in range(int(4+(self.pop_size-4)*0.5),self.dimension):
            sampl = np.random.uniform(low=0, high=1, size=(1,self.dimension))
            a = random.randint(0,3)
            self.pop[i,0:self.dimension]=np.add(self.bestpop[a,0:self.dimension],np.multiply(self.bestpop[a,0:self.dimension],sampl))
            for j in range(0,self.dimension):
                if self.pop[i,j]>=bounds[j,1]: #prevents search space from exceeding the bound (upper limit)
                    self.pop[i,j]=bounds[j,1] 
                if self.pop[i,j]<=bounds[j,0]: #prevents search space from exeeding the bound (lower limit)
                    self.pop[i,j]=bounds[j,0]
    def refresh(self):
        for i in self.exclude:
            self.pop[:,i]=self.x[i]
            
            
    def print_generation(self):
        print(self.pop)
            
        
np.set_printoptions(suppress=True)
        
def GA(bounds,n_particles,max_gen,dimension,exclude,x0,is_online):
    t=time.time()
    pop = Population(n_particles,dimension,x0,exclude,bounds,is_online)
    generation = 0
    hist=[]
    while(generation <= max_gen):
        print("generation: "+str(generation)+":"+str(pop.pop[0,-2]))
        pop.refresh()
        pop.evaluate_gene()
        pop.evaluate_fitness()
        pop.selection()
        pop.crossover()
        pop.mutate()
        hist.append(pop.pop[0,dimension])
        generation +=1
    print(time.time()-t)    
    print("optimisation done")
    plt.plot(hist)
    input1 = ['cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age','compressive strength']
    for i in range(len(input1)):
        print(input1[i]+": "+str(pop.pop[0,i]))


# bounds = np.array([[102,540],[0,359],[0,200],[121,247],[0,32],[801,1145],[594,992],[1,365]])
# x0 =[527, 243, 101, 166, 29, 948, 870, 28]
# exclude =[7]
# GA(bounds,part,iterat,8,exclude,x0)





   
