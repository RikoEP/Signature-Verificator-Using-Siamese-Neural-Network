import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops

from keras import backend as K
from keras.layers import Input, Lambda, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.applications.vgg16 import VGG16


class SiameseNetwork:
    def __init__(self):
        pass

    def extract(self, data):
        path = data.basePath + "\\" + data.name
        print(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # print(img)
        treshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

        dots = treshold > treshold.mean()
        dots_labels = measure.label(dots, background=1)

        image_label_overlay = label2rgb(dots_labels, image=treshold)

        max_area = 0
        total_area = 0
        count_connected_group = 0
        average = 0.0

        for region in regionprops(dots_labels):
            if (region.area > 10):
                total_area += region.area
                count_connected_group += 1

            if (region.area >= 250):
                if (region.area > max_area):
                    max_area = region.area

        average = (total_area / count_connected_group)

        a4_constant = ((average / 84.0) * 250.0) + 100

        b = morphology.remove_small_objects(dots_labels, a4_constant)

        if os.path.isdir(data.basePath + '\\outputs'):
            pass
        else:
            os.mkdir(data.basePath + '\\outputs')

        plt.imsave(data.basePath + '\\outputs\\pre_version.png', b)

        img = cv2.imread(data.basePath + '\\outputs\\pre_version.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cv2.imwrite(data.basePath + "\\outputs\\output.png", img)

        print('Signature Extraction Success!')

        image = cv2.imread(data.basePath + '\\outputs\\output.png')
        result = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([255, 255, 200])
        mask = cv2.inRange(image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 2:
            contours = contours[0]
        else:
            contours = contours[1]

        boxes = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        boxes = np.asarray(boxes)
        left = np.min(boxes[:, 0])
        top = np.min(boxes[:, 1])
        right = np.max(boxes[:, 2])
        bottom = np.max(boxes[:, 3])

        result[close == 0] = (255, 255, 255)
        ROI = result[top:bottom, left:right].copy()

        path = data.basePath + '\\result_'+ data.name
        cv2.imwrite(path, ROI)

        # print('Signature Capture Success')
        return path

    def get_path_list(self, root_path):
        path_list = os.listdir(root_path)

        return path_list

    def get_test_class_names(self, root_path, test_names):
        image_list = []
        image_class_id = []

        for class_image in test_names:
            image_path_list = os.listdir(root_path + '/' + class_image)
            for image_path in image_path_list:
                image_list.append(root_path + '/' + class_image + '/' + image_path)
                image_class_id.append(class_image)

        return image_list, image_class_id

    def get_train_class_names(self, root_path, train_names):
        image_class = []

        for class_image in train_names:
            image_path_list = os.listdir(root_path + '/' + class_image)
            for image_path in image_path_list:
                image_class.append(root_path + '/' + class_image + '/' + image_path)

        return image_class

    def get_train_image_names(self, train_class):
        image_list = []
        image_class_id = []

        for index, image in enumerate(train_class):
            root = train_class[index]
            image_path_list = os.listdir(image)
            image_list.append([root + '/' + image_path_list[0], root + '/' + image_path_list[1]])
            # klo nambah dataset ubah index
            if index < 40:
                image_class_id.append(0)
            else:
                image_class_id.append(1)

        return image_list, image_class_id

    def vgg(self):
        m = VGG16(pooling='avg')
        m.layers.pop()
        m = Model(inputs=m.inputs, outputs=m.layers[-1].output)

        return m

    def siamese(self, input_shape):
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        model = self.vgg()

        encoded_l = model(left_input)
        encoded_r = model(right_input)

        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = Dense(1, activation='sigmoid')(L1_distance)

        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

        return siamese_net

    def verify(self, data):
        test_names = self.get_path_list('.\\static\\signature_data')
        test1_data, label_test = self.get_test_class_names('.\\static\\signature_data', test_names)
        test2_data = data.extraction
        print(data.extraction)

        x1_test = cv2.imread(test2_data)
        x1_test = x1_test.astype('float32') / 255
        x1_test = cv2.resize(x1_test, (224, 224), 3)
        x1_test = x1_test.reshape(1, 224, 224, 3)

        image_test = []
        for i in test1_data:
            x2_test = cv2.imread(i)
            x2_test = x2_test.astype('float32') / 255
            x2_test = cv2.resize(x2_test, (224, 224), 3)
            x2_test = x2_test.reshape(1, 224, 224, 3)

            image_test.append(x2_test)

        input_shape = image_test[0].shape[1:]

        siamese_net = self.siamese(input_shape)
        siamese_net.compile(loss="binary_crossentropy", optimizer=Adam(0.00006))
        siamese_net.load_weights('siamese.h5')

        min = 1000
        id = ''
        for i, image in enumerate(image_test):
            x2_test = image

            image_pair = [x1_test, x2_test]

            loss = siamese_net.predict_on_batch(image_pair)

            print('Signature: {}, Loss: {}'.format(label_test[i], loss))

            if min > loss:
                min = loss
                id = label_test[i]

        # print('Verified Signature: {}'.format(id))
        loss_str = str(min).replace("[","").replace("]","")
        return id,float(loss_str)
