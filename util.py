import numpy as np
import pandas as pd

from bert_embedding import BertEmbedding

CITY='sydney'

def stack(embeddings,stacked=[],n=0):

    if len(stacked)>0: x = stacked
    else: x = embeddings[0]
    
    try:stacked = np.add(np.array(x), embeddings[1])
    except:return np.array(x)
    
    n += 1
    
    return stack(embeddings[1:],stacked=stacked,n=n)



def __bagofBERTs(restaurants,berts):
    
    output = []
    
    for restaurant,(cuisines,embeddings) in zip(restaurants,berts):
        embedding = stack(embeddings)
        output += [(restaurant,cuisines,embedding)]
        
    return output



def getEmbeddings(n_restaurants=100,city=CITY):
    
    print('Cleaning Zomato data for {}.\n'.format(city))
    cuisines, names = zomatoPreprocess(CITY.lower())
    
    print('Retrieving BERT sentence representations for {} restuarants...\n'.format(n_restaurants))
    __bert_embedding = BertEmbedding(model='bert_12_768_12')
    __berts = __bert_embedding(cuisines[:n_restaurants])
    bagofembeddings = __bagofBERTs(names, __berts)
    
    print('Complete.')
    
    filtrd = [(n,c,e) for n,c,e
              in bagofembeddings 
                  if len(e.shape)>0]

    cuisines = [c for n,c,e in filtrd]
    embeds = [e for n,c,e in filtrd]
    names = [n for n,c,e in filtrd]
    
    return names,cuisines,embeds
    

def zomatoPreprocess(city):
    #Zomato preprocessing

    zomato = pd.read_csv('data/'+city+'.csv')
    zomato = zomato[['name','cuisines']].dropna(axis=0)

    cuisines = zomato.cuisines.apply(lambda x:x.replace('[','')
                                                .replace(']','')
                                                .replace('"','')
                                                .replace(',',' ')
                                                .replace('and','') + ' food'
                                                .replace('  ',' '))

    names = zomato.name.values
    
    return cuisines, names