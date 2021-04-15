numFeatures = ["Elevation", "Horizontal_Distance_To_Fire_Points"]
bins = 57
catFeatures = ["Wilderness_Area", "Soil_Type"]


def run(mdh):
    mdh.initialize()
    mdh.rank("table_train", catFeatures, numFeatures, bins).estimate(catFeatures, bins, numFeatures).visualize1D(
        'Wilderness_Area', 'Covertype').predict('table_eval').accuracy()