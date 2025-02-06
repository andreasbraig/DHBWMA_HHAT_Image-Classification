import ssl
from datetime import datetime
import os
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger


def Setup_MobilNet(weights,include_top,Density1,Density2,Density3,Density4,activationx,activationpreds):
  
    #SSL Ausschalten, damit Modell geladen werden kann

    ssl._create_default_https_context = ssl._create_unverified_context  

    base_model = MobileNet(weights=weights, include_top=include_top) # Importing the Base

    #Setup of the Layers for better Results
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(Density1, activationx)(x)
    x = Dense(Density2, activationx)(x)
    x = Dense(Density3, activationx)(x)
    preds = Dense(Density4,activationpreds)(x)

    return base_model, x, preds   

def get_Model(base_model,preds,optimizer,loss,metrics,layers):
    
    model = Model(inputs=base_model.input, outputs=preds)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    for layer in model.layers[:layers]:
        layer.trainable = False
    for layer in model.layers[layers:]:
        layer.trainable = True

    return model

def set_Callbacks(monitor,espatience,RLRPpatience,verbose,factor,min_lr):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = os.path.join(os.getcwd(), "logs")  # Speichert im aktuellen Ordner
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")


    earlystop = EarlyStopping(patience=espatience)

    learning_rate_reduction = ReduceLROnPlateau(monitor=monitor,
                                            patience=RLRPpatience,
                                            verbose=verbose,
                                            factor=factor,
                                            min_lr=min_lr)
    
    csv_logger = CSVLogger(log_file, append=True)

    return [earlystop,learning_rate_reduction,csv_logger]