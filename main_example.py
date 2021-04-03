import sanelib

kmeans = sanelib.kmeans

tablename = "mouse"
feature_names = ["x", "y"]
k = 3
model_identifier = "demo"

model = kmeans.create_model(tablename, feature_names, k, model_identifier)

print(model.get_information())
