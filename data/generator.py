from utils import lobster_list, prepare_json_dataset

numsamples = 30000
backbonelength = 8
p1 = 0.5 # prob. of adding node to a node in backbone
p2 = 0.5 # prob. of adding node to node above
p = 0.5  # prob. of red or blue

lobsterlist= lobster_list(numsamples=numsamples, 
                            backbonelength=backbonelength,
                            p1=p1,
                            p2=p2,
                            p=p)

path = 'dataset'
filename = 'lobsters.json'
filepath = '/'.join([path, filename])
prepare_json_dataset(lobsterlist=lobsterlist, 
                     filepath=filepath)




