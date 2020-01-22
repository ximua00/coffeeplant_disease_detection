import requests
from dataloader import Plant


sample = "C1P19H2.jpg"
resp = requests.post("http://127.0.0.1:5000/predict",
                     files={"file": open('../samples/'+sample,'rb')})

print("Predicted class: {}". format(resp.json()["class_name"]))

plant = Plant(sample)
print("True class: {}". format(resp.json()["class_name"]))


