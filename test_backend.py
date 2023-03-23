import pickle


with open('simulator/ibm_nairobi_backend.pkl', 'rb') as f:
    model = pickle.load(f)[0]


print(model.__dict__)
print(model)