import json
import os
import pycore.preprocessing.imageconversion as uic
import pycore.preprocessing.filemanagement as ufm
import pycore.training.model as tm

#Nomenklatur: 

# u = utilities 
# ic = image
# fm = filemanagement

# ________________ DIESES FILE DIENT DER EINMALIGEN AUSFÜHRUNG ZUM TRAINING EINES MODELLS_______________

#JSON Settings Laden
if __name__ == "__main__":
    config_path = "pycore/setings.json"
    
    cf = json.load(open(config_path, 'r'))
    
    
    #___________Ausführender Bereich
    
    
    #Bilder zu Grayscale
    uic.folder_to_grayscale(cf["filepaths"]["good"],cf["filepaths"]["good_gray"])
    uic.folder_to_grayscale(cf["filepaths"]["bad"],cf["filepaths"]["bad_gray"])
    
    #Aufteilung der Daten
    #Gut
    ufm.split_data(source=cf["filepaths"]["good_gray"],
                   destination1=cf["filepaths"]["test"],
                   destination2=cf["filepaths"]["train"],
                   test_size=cf["ttsplit"]["test_size"],
                   random_state=cf["ttsplit"]["random_state"])
    
    #Schlecht
    ufm.split_data(source=cf["filepaths"]["bad_gray"],
                   destination1=cf["filepaths"]["test"],
                   destination2=cf["filepaths"]["train"],
                   test_size=cf["ttsplit"]["test_size"],
                   random_state=cf["ttsplit"]["random_state"])
    
    #Aufteilung der Train Daten in Train und Validate
    ufm.split_data(source=cf["filepaths"]["train"],
                   destination1=cf["filepaths"]["validate"],
                   destination2=cf["filepaths"]["train"],
                   test_size=cf["ttsplit"]["test_size"],
                   random_state=cf["ttsplit"]["random_state"])
    
    #Funktioniert bis hierhin! 

    train_df = ufm.make_df(os.listdir(cf["filepaths"]["train"]))
    validate_df = ufm.make_df(os.listdir(cf["filepaths"]["validate"]))#
    


    # #aufsetzen des Base Modells

    base,x,preds = tm.Setup_MobilNet(weights= cf["mobilnet"]["weights"],
                                 include_top = cf["mobilnet"]["include_top"],
                                 Density1= cf["mobilnet"]["density1"],
                                 Density2= cf["mobilnet"]["density2"],
                                 Density3= cf["mobilnet"]["density3"],
                                 Density4= cf["mobilnet"]["density4"],
                                 activationpreds= cf["mobilnet"]["activation_preds"],
                                 activationx= cf["mobilnet"]["activation_x"],)

    # Data Augmentation und DataFlowGen

    train_data=ufm.data_gen(mode="train_augmentation",
                            df=train_df,
                            filepath=cf["filepaths"]["train"],
                            x_col="filename",
                            y_col="label",
                            target_size=(cf["imagecfg"]["IMG_W"],cf["imagecfg"]["IMG_H"]),
                            batch_size=cf["batch"]["size"],
                            config=cf,
                            shuffle=True)

    print("train Data Prepared.")
    print(train_data)

    validation_data=ufm.data_gen(mode="validation_augmentation",
                            df=validate_df,
                            filepath=cf["filepaths"]["validate"],
                            x_col="filename",
                            y_col="label",
                            target_size=(cf["imagecfg"]["IMG_W"],cf["imagecfg"]["IMG_H"]),
                            batch_size=cf["batch"]["size"],
                            config=cf,
                            shuffle=True)

    print("validation Data Prepared.")
    print(validation_data)

    # Modell Komplilieren und Einstellung finalisieren

    model = tm.get_Model(base_model=base,
                         preds=preds,
                         optimizer=cf["model_compile"]["optimizer"],
                         loss=cf["model_compile"]["loss"],
                         metrics=[cf["model_compile"]["metrics"]],
                         layers=cf["mobilnet"]["layers"])

    #callbacks

    callbacks = tm.set_Callbacks(monitor=cf["callbacks"]["monitor"],
                                 espatience=cf["callbacks"]["espatience"],
                                 RLRPpatience=cf["callbacks"]["RLRPpatience"],
                                 verbose=cf["callbacks"]["verbose"],
                                 factor=cf["callbacks"]["factor"],
                                 min_lr=cf["callbacks"]["min_lr"])

    # Training 

    batch_size = cf["batch"]["size"]

    epochs=3 if cf["imagecfg"]["FAST_RUN"] else 50
    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=validation_data,
                        validation_steps=validate_df.shape[0]//batch_size,
                        steps_per_epoch=train_df.shape[0]//batch_size,
                        callbacks=callbacks
    )

    model.save("models/mobilenet_v3.h5")