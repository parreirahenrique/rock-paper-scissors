# The example function below keeps track of the opponent"s history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random, os
import tensorflow as tf
import numpy as np
import pandas as pd

def player(prev_play, opponent_history=[], my_history=[]):
    if prev_play == "":
        opponent_history = []
    else:
        opponent_history.append(prev_play)
    
    if len(opponent_history) < 3:
        if prev_play == "":
            my_prev_play = ""

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(1, 12)),
                tf.keras.layers.Dense(36, activation="sigmoid"),
                tf.keras.layers.Dense(3, activation="sigmoid")
            ])

            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            model.save("model.h5")

        else:
            my_prev_play = my_history[-1]

        if (my_prev_play == "P" and prev_play == "R") or (my_prev_play == "R" and prev_play == "S") or (my_prev_play == "S" and prev_play == "P"):
            winner_play = my_prev_play
        elif (prev_play == "P" and my_prev_play == "R") or (prev_play == "R" and my_prev_play == "S") or (prev_play == "S" and my_prev_play == "P"):
            winner_play = prev_play
        elif (my_prev_play == prev_play):
            winner_play = my_prev_play

        if winner_play == "P":
            guess = "S"
        elif winner_play == "R":
            guess = "P"
        elif winner_play == "S":
            guess = "R"
        else:
            guess = random.choice(["P", "R", "S"])

    else:
        if opponent_history[-1] == "P":
            last = 0
            array_last = np.array([1, 0, 0])
        elif opponent_history[-1] == "R":
            last = 1
            array_last = np.array([0, 1, 0])
        elif opponent_history[-1] == "S":
            last = 2
            array_last = np.array([0, 0, 1])

        if opponent_history[-2] == "P":
            array_second_to_last = np.array([1, 0, 0])
        elif opponent_history[-2] == "R":
            array_second_to_last = np.array([0, 1, 0])
        elif opponent_history[-2] == "S":
            array_second_to_last = np.array([0, 0, 1])

        if opponent_history[-3] == "P":
            array_third_to_last = np.array([1, 0, 0])
        elif opponent_history[-3] == "R":
            array_third_to_last = np.array([0, 1, 0])
        elif opponent_history[-3] == "S":
            array_third_to_last = np.array([0, 0, 1])

        if my_history[-1] == "P":
            my_array_last = np.array([1, 0, 0])
        elif my_history[-1] == "R":
            my_array_last = np.array([0, 1, 0])
        elif my_history[-1] == "S":
            my_array_last = np.array([0, 0, 1])

        if my_history[-2] == "P":
            my_array_second_to_last = np.array([1, 0, 0])
        elif my_history[-2] == "R":
            my_array_second_to_last = np.array([0, 1, 0])
        elif my_history[-2] == "S":
            my_array_second_to_last = np.array([0, 0, 1])

        if my_history[-3] == "P":
            my_array_third_to_last = np.array([1, 0, 0])
        elif my_history[-3] == "R":
            my_array_third_to_last = np.array([0, 1, 0])
        elif my_history[-3] == "S":
            my_array_third_to_last = np.array([0, 0, 1])
        
        row_train = np.concatenate((my_array_third_to_last, my_array_second_to_last, array_third_to_last, array_second_to_last)).reshape(1, 1, 12)
        row_prediction = np.concatenate((my_array_second_to_last, my_array_last, array_second_to_last, array_last)).reshape(1, 1, 12)
        label_train = np.array(last).reshape(1, 1, 1)
        
        model = tf.keras.models.load_model("model.h5")
        
        model.fit(row_train, label_train, epochs=100, verbose=0)
        
        prediction = model.predict(row_prediction)
        pred = ["P", "R", "S"][np.argmax(prediction)]

        model.save("model.h5")
        
        if pred == "P":
            guess = "S"
        elif pred == "R":
            guess = "P"
        elif pred == "S":
            guess = "R"
    
    my_history.append(guess)

    return guess
