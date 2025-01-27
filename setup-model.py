import json
import os
import pycore.preprocessing.filemanagement as ufm
import pycore.training.model as tm

config_path = "pycore/setings.json"

cf = json.load(open(config_path, 'r'))

#Erstellung der Dataframes für Training und Validation

train_df = ufm.make_df(os.listdir(cf["filepaths"]["train"]))
validate_df = ufm.make_df(os.listdir(cf["filepaths"]["validate"]))


#aufsetzen des Base Modells

base,x,preds = tm.Setup_MobilNet(weights= cf["mobilnet"]["weights"],
                                 include_top = cf["mobilnet"]["include_top"],
                                 Density1= cf["mobilnet"]["density1"],
                                 Density2= cf["mobilnet"]["density2"],
                                 Density3= cf["mobilnet"]["density3"],
                                 Density4= cf["mobilnet"]["density4"],
                                 activationpreds= cf["mobilnet"]["activation_x"],
                                 activationx= cf["mobilnet"]["activation_preds"],)


# Vorbereitung für Training 

