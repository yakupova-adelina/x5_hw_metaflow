from metaflow import FlowSpec, step

import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

tf.random.set_seed(1234)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


class PointClassificationFlow(FlowSpec):

    @step
    def start(self):

        DATA_DIR = tf.keras.utils.get_file(
            "modelnet.zip",
            "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
            extract=True,
        )
        DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

        print('Data loaded!')

        def parse_dataset(num_points=2048):
            train_points = []
            train_labels = []
            test_points = []
            test_labels = []
            class_map = {}
            folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

            for i, folder in enumerate(folders):
                print("processing class: {}".format(os.path.basename(folder)))
                # store folder name with ID so we can retrieve later
                class_map[i] = folder.split("/")[-1]
                # gather all files
                train_files = glob.glob(os.path.join(folder, "train/*"))
                test_files = glob.glob(os.path.join(folder, "test/*"))

                for f in train_files:
                    train_points.append(trimesh.load(f).sample(num_points))
                    train_labels.append(i)

                for f in test_files:
                    test_points.append(trimesh.load(f).sample(num_points))
                    test_labels.append(i)

            return (
                np.array(train_points),
                np.array(test_points),
                np.array(train_labels),
                np.array(test_labels),
                class_map,
            )

        self.num_points = 2048
        self.num_classes = 10
        self.batch_size = 32

        self.train_points, self.test_points, self.train_labels, \
        self.test_labels, self.CLASS_MAP = parse_dataset(self.num_points)
        self.next(self.data_augment)

    @step
    def data_augment(self):

        def augment(points, label):
            # jitter points
            points += tf.random.uniform(points.shape, -0.005, 0.005,
                                        dtype=np.float64)
            # shuffle points
            points = tf.random.shuffle(points)
            return points, label

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_points, self.train_labels))
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.test_points, self.test_labels))
        self.train_dataset = self.train_dataset.shuffle(
            len(self.train_points)
        ).map(augment).batch(self.batch_size)
        self.test_dataset = self.test_dataset.shuffle(
            len(self.test_points)
        ).batch(self.batch_size)

        self.next(self.build_model)

    @step
    def build_model(self):

        def conv_bn(x, filters):
            x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
            x = layers.BatchNormalization(momentum=0.0)(x)
            return layers.Activation("relu")(x)

        def dense_bn(x, filters):
            x = layers.Dense(filters)(x)
            x = layers.BatchNormalization(momentum=0.0)(x)
            return layers.Activation("relu")(x)

        def tnet(inputs, num_features):
            # Initalise bias as the indentity matrix
            bias = keras.initializers.Constant(np.eye(num_features).flatten())
            reg = OrthogonalRegularizer(num_features)

            x = conv_bn(inputs, 32)
            x = conv_bn(x, 64)
            x = conv_bn(x, 512)
            x = layers.GlobalMaxPooling1D()(x)
            x = dense_bn(x, 256)
            x = dense_bn(x, 128)
            x = layers.Dense(
                num_features * num_features,
                kernel_initializer="zeros",
                bias_initializer=bias,
                activity_regularizer=reg,
            )(x)
            feat_T = layers.Reshape((num_features, num_features))(x)
            # Apply affine transformation to input features
            return layers.Dot(axes=(2, 1))([inputs, feat_T])

        inputs = keras.Input(shape=(self.num_points, 3))

        x = tnet(inputs, 3)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)

        self.outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = keras.Model(
            inputs=inputs,
            outputs=self.outputs,
            name="pointnet")
        self.model.summary()
        self.next(self.train)

    @step
    def train(self):
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=["sparse_categorical_accuracy"],
        )

        self.model.fit(
            self.train_dataset,
            epochs=20,
            validation_data=self.test_dataset
        )
        self.next(self.visualize)

    @step
    def visualize(self):
        data = self.test_dataset.take(1)

        points, labels = list(data)[0]
        points = points[:8, ...]
        labels = labels[:8, ...]

        # run test data through model
        preds = self.model.predict(points)
        preds = tf.math.argmax(preds, -1)

        points = points.numpy()

        # plot points with predicted class and label
        fig = plt.figure(figsize=(15, 10))
        for i in range(8):
            ax = fig.add_subplot(2, 4, i + 1, projection="3d")
            ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
            ax.set_title(
                "pred: {:}, label: {:}".format(
                    self.CLASS_MAP[preds[i].numpy()],
                    self.CLASS_MAP[labels.numpy()[i]]
                )
            )
            ax.set_axis_off()
        plt.show()
        self.next(self.end)

    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    PointClassificationFlow()
