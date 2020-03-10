import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import matplotlib.gridspec as gridspec

def var_info(v):
    if not isinstance(v, str):
        print("The function var_info takes the name of the variable as a string!")
        sys.exit(0)
    print("{} is of type {} with shape {}".format(v, eval("type({})".format(v)), eval("{}.shape".format(v))))


def scoop_and_preprocess_data():
    global fashion_data, x_train, x_test, y_train, y_test, target_set
    global labels

    # labels scooped from https://github.com/zalandoresearch/fashion-mnist
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # grab fashion mnist data - in some custome class

    fashion_data = tf.keras.datasets.fashion_mnist



    # pull out train and test in/out numpy arrays from class

    (x_train, y_train), (x_test, y_test) = fashion_data.load_data()

    # get feel for size and shape of things

    var_info("x_train")
    var_info("y_train")

    target_set = set(y_train)

    print("Num unique targets appears to be {}".format(len(target_set)))
    print(target_set)

    # get inputs to be the right shape/dimensionality - we need a colour dimension (which happens to be of size 1 cos B&W)

    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test,-1)

    var_info("x_train")


def build_and_compile_model():
    global net_design
    # build network (model?) - a few cnn layers then normal layers

    inp_layer = tf.keras.layers.Input(shape=x_train[0].shape)
    net_layers = tf.keras.layers.Conv2D(29, (3, 3), activation="relu")(inp_layer)
    net_layers = tf.keras.layers.Conv2D(68, (3, 3), activation="relu")(inp_layer)
    net_layers = tf.keras.layers.Conv2D(108, (3, 3), activation="relu")(inp_layer)
    net_layers = tf.keras.layers.Flatten()(net_layers)
    net_layers = tf.keras.layers.Dense(30, activation="relu")(net_layers)
    net_layers = tf.keras.layers.Dense(len(target_set), activation="softmax")(net_layers)

    net_design = tf.keras.models.Model(inp_layer, net_layers)

    # compile: i.e. specify the learing mechanism - grad desc and error func sparce-cat-x-ent

    net_design.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


def do_the_learning():
    # learn! (fit)... specify data and epochs

    net_design.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)


def show_an_input_output_pair(xarr, yarr, netout):
    n = xarr.shape[0]
    rows = 1+int(n/5)
    cols = 5

    fig = plt.figure()
    fig.suptitle("A collection of {} images".format(n))

    gridspec_array = fig.add_gridspec(rows*2, cols)

    for i, (x, y, nout) in enumerate(zip(xarr, yarr, netout)):
        r = int(i/5) * 2
        c = i % 5
        ax = fig.add_subplot(gridspec_array[r, c])
        image_without_the_extra_colour_dimension = np.reshape(x, (28, 28))
        plt.xlabel("{}: {}".format(y, labels[y]))
        plt.xticks([])
        plt.yticks([])
        ax.imshow(image_without_the_extra_colour_dimension, cmap="gray")

        ax = fig.add_subplot(gridspec_array[r+1, c])
        targ = np.zeros((10,))
        targ[y] = 1
        ax.bar(list(range(10)), targ, color="red")
        ax.bar(list(range(10)), nout, color="green")
        plt.xticks(list(range(10)), labels=list(range(10)))
        plt.yticks([])
        ax.set_ylim([0, 1])
        ax.set_xlim([-.5, 9.5])


    plt.show()

scoop_and_preprocess_data()

show_n = 16
dummie_out = np.zeros((show_n,len(target_set)))
dummie_out[0,0] = .01
dummie_out[0,3] = 1.0
dummie_out[0,7] = .5
dummie_out[0,9] = .1
#how_an_input_output_pair(x_train[:show_n],y_train[:show_n], dummie_out)

scratch = input("Learn from scratch?")
# scratch = "n"

if scratch[:1] == "y":
    build_and_compile_model()
    do_the_learning()
    net_design.save('fashion_model.h5')
else:
    net_design = tf.keras.models.load_model('fashion_model.h5')

predicted_answer_as_10_floats = net_design.predict(x_test)
show_an_input_output_pair(x_test[:show_n], y_test[:show_n], predicted_answer_as_10_floats)
