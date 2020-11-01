import pickle


class Test:
    def __init__(self):
        pass


top_sorted_fi = Test()
fn = 'pretrained_features.pkl'
with open(fn, 'wb') as f:  # open file with write-mode
    picklestring = pickle.dump(top_sorted_fi, f)  #

with open(fn, 'rb') as f:
    obj = pickle.load(f)
print("obj:", obj)
