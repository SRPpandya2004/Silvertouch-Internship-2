from DenguePrediction_Baseclass import  Multi_Output, Iterative_Multistep, Backshift_Transformation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from fastapi import FastAPI, HTTPException
import os
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, LeakyReLU, ELU
plt.style.use("fivethirtyeight")
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format

# We make to list to store the Hyperparameters and result into csv
LSTM_Multioutput = []
LSTM_Iterative_Multistep = []
LSTM_backshift_transformation_result = []



# this class does the forecasting with Multi Ouput method
class LSTMModel_Multi_Output(Multi_Output):

    # we Hypertune the model with the help Optuna
    def hypertuning(self):
        
        # Hypertuning
        def objective(trial):
            window_length = trial.suggest_int('window_length', 1, 15)
            neurons = trial.suggest_int('neurons', 32, 256, log=True)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
            epochs = trial.suggest_int('epochs', 10, 50)
            batch_size = trial.suggest_categorical('batch_size',[5, 10, 16, 32, 64])
            optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'])
            activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'elu'])
            early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 20)
            
            # Train test split
            num_features = len(self.df.columns)
            test_size = 0.2

            # Making Train Dataset
            training_dataset = self.scaled_df[:-int(test_size * len(self.scaled_df))]
            X_train, y_train = [], []
            for i in range(window_length, len(training_dataset) - self.forecasting_stamp):
                
                # We have to put condition because the univariate and multivariate have different shapes
                if self.num_features == 1:
                    X_train.append(training_dataset[i-window_length:i, num_features - 1])
                else:
                    X_train.append(training_dataset[i-window_length:i, :num_features - 1])
                y_train.append(training_dataset[i:i+self.forecasting_stamp, num_features - 1])


            # Making Test Dataset
            test_dataset = self.scaled_df[-int(test_size * len(self.scaled_df))-window_length:]
            X_test, y_test = [],[]
            y_test = test_dataset[window_length:, num_features - 1]
            for i in range(window_length, (len(test_dataset) - self.forecasting_stamp), self.forecasting_stamp):

                # We have to put condition because the univariate and multivariate have different shapes
                if self.num_features == 1:
                    X_test.append(test_dataset[i - window_length:i, num_features - 1])
                else:
                    X_test.append(test_dataset[i - window_length:i, :num_features - 1])


            # Converting them into numpy arrays
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_test, y_test = np.array(X_test), np.array(y_test)

            # Check if X_train's dimension is < 3.
            if X_train.ndim < 3:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


            # Build
            model = Sequential()
            if activation == 'leaky_relu':
                model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), activation=LeakyReLU(alpha=0.01)))
            elif activation == 'elu':
                model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), activation=ELU(alpha=1.0)))
            else :
                model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), activation=activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(self.forecasting_stamp))

            # compile
            model.compile(optimizer = optimizer, loss='mean_squared_error', metrics = ['accuracy'])
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience= early_stopping_patience,
                restore_best_weights=True
            )
            # fitting
            model.fit(X_train, y_train, epochs= epochs, steps_per_epoch = int(len(X_train) / batch_size), batch_size = batch_size, callbacks=[early_stopping], shuffle= False)
            model.save(f'./pickles/LSTM_pickles/LSTM_model_multioutput_{self.pickle_index}.pkl')
            # Making predictions
            predictions = model.predict(X_test)
            # Reshaping and Inverse Transforming the values
            predictions = np.reshape(predictions, (predictions.shape[1] * predictions.shape[0], 1))
            df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(test_dataset[-(len(predictions)): , :num_features - 1])], axis=1)
            rev_trans = self.scaler.inverse_transform(df_pred)

            # final DataFrame
            pd.set_option('display.max_rows', None)
            final_df = self.df[-(len(predictions)):]
            final_df['predictions'] = rev_trans[:,0]
            RMSE = np.sqrt(mean_squared_error(final_df['predictions'], final_df['total_cases']))
            return RMSE

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials= 5)
        self.best_trial = study.best_trial
        print(f"the Best parameters : {self.best_trial.params}\n Best score is {self.best_trial.value}")

    # We split the data into training and testing data
    def train_test_split(self, test_size):
        try:
            # if hasattr(self, 'best_trial'):
            self.window_length = self.best_trial.params['window_length']
            # else:
            #     raise AttributeError("'best_trial' not found")
        except Exception as e:
            # Load the model if the pickle file exists and extract window_length from the model
           
            model_path = f"./pickles/LSTM_pickles/LSTM_model_multioutput_{self.pickle_index}.h5"
            if os.path.exists(model_path):
                # loaded_model = joblib.load(pickle_path)
                loaded_model = tf.keras.models.load_model(model_path)
               
                input_layer = loaded_model.input_shape[1]
                
                self.window_length = input_layer
            else:
                self.window_length = 1
            
            print("Error encountered:", e)
        super().train_test_split(test_size)

    # Building and training the model 
    def building_model(self):
        try:
            self.neurons = self.best_trial.params['neurons']
            self.dropout_rate = self.best_trial.params['dropout_rate']
            self.epochs = self.best_trial.params['epochs']
            self.batch_size = self.best_trial.params['batch_size']
            self.optimizer = self.best_trial.params['optimizer']
            self.activation = self.best_trial.params['activation']
            self.early_stopping_patience = self.best_trial.params['early_stopping_patience']
        except Exception as e:
            print("into the exception of lstm output", e)
            self.neurons = 177
            self.dropout_rate = 0.1
            self.epochs = 36
            self.batch_size = 32
            self.optimizer = 'adam'
            self.activation = 'leaky_relu'
            self.early_stopping_patience = 12

        # Building the model
        model = Sequential()
        if self.activation == 'leaky_relu':
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=LeakyReLU(alpha=0.01)))
        elif self.activation == 'elu':
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=ELU(alpha=1.0)))
        else:
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.forecasting_stamp))

        # Compile
        model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['accuracy'])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )
        
        # Fitting the model
        model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stopping], shuffle=False)
        
        # Here we make the folder/directory if doesn't exist and specify the paths
        pickle_dir = "./pickles/LSTM_pickles"
        os.makedirs(pickle_dir, exist_ok=True)
        model_path = os.path.join(pickle_dir, f"LSTM_model_multioutput_{self.pickle_index}.h5")
        model.save(model_path)
        
        return model


    # Testing and evaluating the model
    def Test(self):
        try:
            # self.model =  joblib.load(f"./pickles/LSTM_pickles/LSTM_model_multioutput_{self.pickle_index}.pkl")
            model_path = f"./pickles/LSTM_pickles/LSTM_model_multioutput_{self.pickle_index}.h5"
       
            self.model = tf.keras.models.load_model(model_path)
            
            predictions,MSE, RMSE, MAE, R2 = super().Test()
           
            print(f"MSE: {MSE}, RMSE: {RMSE}, MAE: {MAE}, R2: {R2}")
            return predictions,MSE, RMSE, MAE, R2
        except Exception as e:
            print("except", e)
            raise HTTPException(status_code=500, detail=str(e))
    # forecasting function which is being used specificly in the backshift transformation method
    def plot_results(self):

        plot_dir = "./Plots/LSTM_MultiOutput_Plots"  # Specify the directory to save plots
        os.makedirs(plot_dir, exist_ok=True)
        plot_path_1 = os.path.join(plot_dir, f"LSTM_MultiOutput_plot_{self.pickle_index}.png")
        plot_path_2 = os.path.join(plot_dir, f"Other_LSTM_MultiOutput_plot_{self.pickle_index}.png")
        super().plot_results(plot_path_1, plot_path_2)

    # We make to list to store the Hyperparameters and result into csv
    def store_results(self):
        # Keeping the best results in the csv file
        LSTM_Multioutput.append({
                'Model' : "LSTM Multi output logic",
                'Features': self.feature_list,
                'Window Length' : self.best_trial.params['window_length'],
                'Neurons': self.best_trial.params['neurons'],
                'Dropout rate' : self.best_trial.params['dropout_rate'],
                'Epochs': self.best_trial.params['epochs'],
                'Batch Size' : self.best_trial.params['batch_size'],
                'Optimizer' : self.best_trial.params['optimizer'],
                'Activation function' : self.best_trial.params['activation'],
                'Early stopping patience' : self.best_trial.params['early_stopping_patience'],
                'RMSE': self.best_trial.value
            })

        # Save results to a CSV file
        result_df = pd.DataFrame(LSTM_Multioutput)
        result_df.to_csv("./Data/LSTM_Multioutput.csv", index=False)


# this class does the forecasting with Iterative Multistep method
class LSTMModel_Iterative_Multistep(Iterative_Multistep):
    
    # we Hypertune the model with the help Optuna
    def hypertuning(self):
        # Hypertuning
        def objective(trial):
            window_length = trial.suggest_int('window_length', 1, 15)
            neurons = trial.suggest_int('neurons', 32, 256, log=True)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
            epochs = trial.suggest_int('epochs', 10, 50)
            batch_size = trial.suggest_categorical('batch_size',[5, 10, 16, 32, 64])
            optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'])
            activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'elu'])
            early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 20)
            
            
            # how many stamps we wanna predict into the future
            window_length = window_length
            self.num_features = len(self.df.columns)
            test_size = 0.2
            # Making Train Dataset
            training_dataset = self.scaled_df[:-int(test_size * len(self.scaled_df)-window_length)]
            x_train, y_train = [], []
            for i in range(window_length, len(training_dataset)):
                x_train.append(training_dataset[i-window_length:i,  :self.num_features]) # independent variabels
                y_train.append(training_dataset[i, :self.num_features]) # Target variable
            # Making Test Dataset
            test_dataset = self.scaled_df[-int(test_size * len(self.scaled_df))-window_length:]
            x_test, y_test = [],[]
            y_test = test_dataset[window_length:, :self.num_features]
            for i in range(window_length, (len(test_dataset)), self.forecasting_stamp):
                x_test.append(test_dataset[i - window_length:i, :self.num_features])
            # Converting them into numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_test, y_test = np.array(x_test), np.array(y_test)
            print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
            

            # Build
            model = Sequential()
            if activation == 'leaky_relu':
                model.add(LSTM(neurons, input_shape=(x_train.shape[1], x_train.shape[2]), activation=LeakyReLU(alpha=0.01)))
            elif activation == 'elu':
                model.add(LSTM(neurons, input_shape=(x_train.shape[1], x_train.shape[2]), activation=ELU(alpha=1.0)))
            else :
                model.add(LSTM(neurons, input_shape=(x_train.shape[1], x_train.shape[2]), activation=activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(self.num_features))
            
            # Compile
            model.compile(optimizer= optimizer, loss=['mean_squared_error'], metrics=['accuracy'])
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience= early_stopping_patience,
                restore_best_weights=True
            )
            # fitting the model
            model.fit(x_train, y_train, epochs= epochs, steps_per_epoch = int(len(x_train) / batch_size), batch_size = batch_size, callbacks=[early_stopping], shuffle= False)                
            
            # Prediction and Evaluation
            predictions = np.zeros(shape=(len(y_test), 1))
            count = 0
            for i in range(x_test.shape[0]):
                x_temp = x_test[i:i+1, :, :] # if we use slicing while selectng rows then we get multi-dimentional output
                for j in range(self.forecasting_stamp):
                    if x_temp.ndim < 3:
                        x_temp = x_temp.reshape(1, x_temp.shape[0], x_temp.shape[1])
                    if count >= len(y_test):
                        break
                    prediction = model.predict(x_temp[:, -window_length:, :])
                    # we have to convert it into 3D shape because we have append it in the x_temp which is 3D.
                    prediction = prediction.reshape(1, prediction.shape[0], prediction.shape[1])
                    print(prediction.shape, prediction.ndim)
                    x_temp = np.append(x_temp, prediction[:,:,:self.num_features], axis=1)
                    predictions[count] = prediction[:,:,self.num_features-1]
                    count += 1

                # Inverse transforming it
                df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(test_dataset[-len(predictions):,:self.num_features - 1])], axis=1)
                rev_trans = self.scaler.inverse_transform(df_pred)

                # making a dataframe
                final_df = self.df[-len(predictions):]
                final_df['predictions'] = rev_trans[:, 0]
            RMSE = np.sqrt(mean_squared_error(final_df['predictions'], final_df['total_cases']))
            return RMSE
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials= 5)
        self.best_trial = study.best_trial
        print(f"Best parameters : {self.best_trial.params}\n Best score is {self.best_trial.value}")

    # We split the data into training and testing data
    def train_test_split(self, test_size):
        try:
            self.window_length = self.best_trial.params['window_length']
        except Exception as e:
            print(e)
            # we Load the model if the pickle file exists and exctract the window_length from the model
            try :
                if os.path.exists(f"./pickles/LSTM_pickles/LSTM_model_Iterative_multistep_{self.pickle_index}.pkl"):
                    loaded_model = joblib.load(f"./pickles/LSTM_pickles/LSTM_model_Iterative_multistep_{self.pickle_index}.pkl")
                    input_layer = loaded_model.input_shape[1]
                    self.window_length = input_layer
            except Exception as e1:
                print(e1)
                self.window_length = 7

        super().train_test_split(test_size)

    # Building and training the model 
    def building_model(self):
        # We insert exception handling just in case this function is called without hypertuning
        try:
            self.neurons = self.best_trial.params['neurons']
            self.dropout_rate = self.best_trial.params['dropout_rate']
            self.epochs = self.best_trial.params['epochs']
            self.batch_size = self.best_trial.params['batch_size']
            self.optimizer = self.best_trial.params['optimizer']
            self.activation = self.best_trial.params['activation']
            self.early_stopping_patience = self.best_trial.params['early_stopping_patience']

        except Exception as e:
            print(e)
            self.neurons = 177
            self.dropout_rate = 0.1
            self.epochs = 36
            self.batch_size = 32
            self.optimizer = 'adam'
            self.activation = 'leaky_relu'
            self.early_stopping_patience = 12

        model = Sequential()
        if self.activation == 'leaky_relu':
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=LeakyReLU(alpha=0.01)))
        elif self.activation == 'elu':
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=ELU(alpha=1.0)))
        else :
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_features))
        
        # Compile
        model.compile(optimizer= self.optimizer, loss=['mean_squared_error'], metrics=['accuracy'])

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience= self.early_stopping_patience,
            restore_best_weights=True
        )
        # fitting the model
        model.fit(self.X_train, self.y_train, epochs= self.epochs, steps_per_epoch = int(len(self.X_train) / self.batch_size), batch_size = self.batch_size, callbacks=[early_stopping], shuffle= False)
        # Here we make the folder/directory if doesn't exist and specify the paths
        pickle_dir = "./pickles/LSTM_pickles"  # Specify the directory to save pickels
        os.makedirs(pickle_dir, exist_ok=True)
        self.pickle_path = os.path.join(pickle_dir, f"LSTM_model_Iterative_multistep_{self.pickle_index}.pkl")
        joblib.dump(model, self.pickle_path) # Dump the pickle file with the help of joblib
        return model

    # Testing and evaluating the model
    def Test(self):
        # Loading the pickle file
        self.model =  joblib.load(f"./pickles/LSTM_pickles/LSTM_model_Iterative_multistep_{self.pickle_index}.pkl")
        predictions,MSE, RMSE, MAE, R2 = super().Test()
        return predictions, MSE, RMSE, MAE, R2 

    # we Plot the results to visualize and understand them better
    def plot_results(self):
        plot_dir = "./Plots/LSTM_Iterative_Multistep_Plots"  # Specify the directory to save plots
        os.makedirs(plot_dir, exist_ok=True)
        plot_path_1 = os.path.join(plot_dir, f"LSTM_Iterative_Multistep_Plot_{self.pickle_index}.png")
        plot_path_2 = os.path.join(plot_dir, f"Other_LSTM_Iterative_Multistep_Plot_{self.pickle_index}.png")
        super().plot_results(plot_path_1, plot_path_2)
   
    # We make to list to store the Hyperparameters and result into csv
    def store_results(self):
         # Keeping the best results in the csv file
        LSTM_Iterative_Multistep.append({
                'Model' : "LSTM Iterative Multistep logic",
                'Features': self.feature_list,
                'Window Length' : self.best_trial.params['window_length'],
                'Neurons': self.best_trial.params['neurons'],
                'Dropout rate' : self.best_trial.params['dropout_rate'],
                'Epochs': self.best_trial.params['epochs'],
                'Batch Size' : self.best_trial.params['batch_size'],
                'Optimizer' : self.best_trial.params['optimizer'],
                'Activation function' : self.best_trial.params['activation'],
                'Early stopping patience' : self.best_trial.params['early_stopping_patience'],
                'RMSE': self.best_trial.value
            })

        # Save results to a CSV file
        LSTM_Optimized_df = pd.DataFrame(LSTM_Iterative_Multistep)
        LSTM_Optimized_df.to_csv("./Data/LSTM_Iterative_Multistep.csv", index=False)


# this class does the forecasting with Backshift Transformation method
class LSTMModel_Backshift_Transformation(Backshift_Transformation):

    # We preprocess the data in this function, like removing outliers and handling missing values and such
    def preprocess_data(self, forecasting_stamp):
        # Calling the method from parent class
        super().preprocess_data(forecasting_stamp)
        super().preprocess_data_backshift_transformation()

    # we Hypertune the model with the help Optuna
    def hypertuning(self):
        # Optuna
        def create_model(neurons, dropout_rate, activation):
            model = Sequential()
            if activation == 'leaky_relu':
                model.add(LSTM(neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=LeakyReLU(alpha=0.01)))
            elif activation == 'elu':
                model.add(LSTM(neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=ELU(alpha=1.0)))
            else :
                model.add(LSTM(neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation= activation))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1))
            return model

        def objective(trial):
            
            # Define the search space for hyperparameters
            neurons = trial.suggest_int('neurons', 32, 256, log=True)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
            epochs = trial.suggest_int('epochs', 10, 50)
            batch_size = trial.suggest_categorical('batch_size',[5, 10, 16, 32, 64])
            optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'])
            activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'elu'])
            early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 20)

            # Creating the model
            model = create_model(neurons, dropout_rate, activation)
            # Compile the model
            model.compile(optimizer = 'adam', loss='mean_squared_error', metrics = ['accuracy'])
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience= early_stopping_patience,
                restore_best_weights=True
            )
            # Train the model
            model.fit(self.X_train, self.y_train, batch_size= batch_size, epochs = epochs,callbacks=[early_stopping], shuffle = False)
            
            # Prediction
            predictions = model.predict(self.X_test)
            df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(self.X[-len(predictions):,:self.num_features - 1])], axis=1)
            rev_trans = self.scaler.inverse_transform(df_pred)
            final_df = self.df[-len(predictions):]
            final_df['predictions'] = rev_trans[:, 0]

            RMSE = np.sqrt(mean_squared_error(final_df['predictions'], final_df['total_cases']))
            return RMSE

        # Creating the study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials= 5)

        # Print the best hyperparameters found
        self.best_trial = study.best_trial
        print(f"the Best Trial:  Loss : {self.best_trial.value}, Params : {self.best_trial.params}")

    # Building and training the model 
    def building_model(self):

        try:
            self.neurons = self.best_trial.params['neurons']
            self.dropout_rate = self.best_trial.params['dropout_rate']
            self.epochs = self.best_trial.params['epochs']
            self.batch_size = self.best_trial.params['batch_size']
            self.optimizer = self.best_trial.params['optimizer']
            self.activation = self.best_trial.params['activation']
            self.early_stopping_patience = self.best_trial.params['early_stopping_patience']

        except Exception as e:
            print(e)
            self.neurons = 177
            self.dropout_rate = 0.1
            self.epochs = 36
            self.batch_size = 32
            self.optimizer = 'adam'
            self.activation = 'leaky_relu'
            self.early_stopping_patience = 12


        model = Sequential()
        if self.activation == 'leaky_relu':
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=LeakyReLU(alpha=0.01)))
        elif self.activation == 'elu':
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=ELU(alpha=1.0)))
        else :
            model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), activation=self.activation))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer= 'adam', loss=['mean_squared_error'], metrics= ['accuracy'])

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience= self.early_stopping_patience,
            restore_best_weights=True
        )

        # Fit the model
        model.fit(self.X_train, self.y_train, epochs= self.epochs, batch_size= self.batch_size,callbacks= [early_stopping], shuffle= False)

        # Making a pickle file
        # Here we make the folder/directory if doesn't exist and specify the paths
        pickle_dir = "./pickles/LSTM_pickles"  # Specify the directory to save pickels
        os.makedirs(pickle_dir, exist_ok=True)
        self.pickle_path = os.path.join(pickle_dir, f"LSTM_backshiftTransform_model_{self.pickle_index}.pkl")
        joblib.dump(model, self.pickle_path) # Dump the pickle file with the help of joblib
        return model

    # Testing and evaluating the model
    def Test(self):
        # Loading the pickle file
        self.model = joblib.load(f"./pickles/LSTM_pickles/LSTM_backshiftTransform_model_{self.pickle_index}.pkl")
        predictions,MSE, RMSE, MAE, R2 = super().Test()
        return predictions, MSE,  RMSE, MAE, R2

    # we Plot the results to visualize and understand them better
    def plot_results(self):
        # Here we make the folder if doesn't exist for the plots and specify the paths
        plot_dir = "./Plots/LSTM_backshiftTransform_plots"  # Specify the directory to save plots
        os.makedirs(plot_dir, exist_ok=True)
        plot_path_1 = os.path.join(plot_dir, f"LSTM_backshift_tranformation_plot_{self.pickle_index}.png")
        plot_path_2 = os.path.join(plot_dir, f"Other_LSTM_backshift_tranformation_plot_Test_{self.pickle_index}.png")
        super().plot_results(plot_path_1, plot_path_2)

    # We make to list to store the Hyperparameters and result into csv
    def store_results(self):
        # Keeping the best results in the csv file
        LSTM_backshift_transformation_result.append({
                'Model' : "LSTM Backshift Transformation logic",
                'Features': self.feature_list,
                'Neurons': self.best_trial.params['neurons'],
                'Dropout rate' : self.best_trial.params['dropout_rate'],
                'Epochs': self.best_trial.params['epochs'],
                'Batch Size' : self.best_trial.params['batch_size'],
                'Optimizer' : self.best_trial.params['optimizer'],
                'Activation function' : self.best_trial.params['activation'],
                'Early stopping patience' : self.best_trial.params['early_stopping_patience'],
                'RMSE': self.best_trial.value
            })
            
        # Save results to a CSV file
        LSTM_backshift_transformation = pd.DataFrame(LSTM_backshift_transformation_result)
        LSTM_backshift_transformation.to_csv("./Data/LSTM_backshift_transformation_result.csv", index=False)

