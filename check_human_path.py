import pickle

with open('./data/human_path.pkl', 'rb') as f:
  a = pickle.load(f)
  
print(a)
print(a.keys())
