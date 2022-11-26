import json
for i in range(5):
    with open("inputdata/five_fold_"+str(i+1)+"_test.json", 'rb') as f:
        D = json.load(f)
    d1 = dict(D)
    json_object = json.dumps(d1, indent=4)
    with open("five_fold_"+str(i+1)+"_test.json", "w") as outfile:
        outfile.write(json_object)
