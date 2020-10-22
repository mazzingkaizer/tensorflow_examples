from disable_warnings import *
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
from tools import send_feedback, print_stderr
import converter


if __name__ == "__main__":
    try:
        part_id = sys.argv[2]
    except IndexError:
        print_stderr("Missing partId. Required to continue.")
        send_feedback(0.0, "Missing partId.")
        exit()
    else:
        if part_id != "wNSsr":
            print_stderr("Invalid partId. Required to continue.")
            send_feedback(0.0, "Invalid partId.")
            exit()



try:
    student_model = tf.saved_model.load("./tmp/mymodel/1")
except:
    send_feedback(0.0, "Your model could not be loaded. Make sure the zip file has the correct contents.")

infer = student_model.signatures["serving_default"]

splits = ["train[:80%]", "train[80%:90%]", "train[90%:]"]
(train_examples, validation_examples, test_examples), info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True, split=splits, data_dir='./data')
num_examples = info.splits["train"].num_examples
num_classes = info.features["label"].num_classes


def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label


test_batches = test_examples.map(format_image).batch(1)


eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="eval_accuracy")
for images, labels in test_batches:
    try:
        predictions = infer(images)["output_1"]
    except:
        send_feedback(0.0, "There was an issue with your model that prevented inference.")
    eval_accuracy(labels, predictions)

score = (eval_accuracy.result() * 100).numpy()

if score > 60:
    send_feedback(1.0, "Congratulations! Your model achieved the desired level of accuracy.")
else:
    send_feedback(0.0, "Your model has an accuracy lower than 0.6.")
