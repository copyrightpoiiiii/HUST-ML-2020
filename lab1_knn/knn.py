import numpy as np
import struct
import matplotlib.pyplot as plt
import re

# 训练集文件
train_images_idx3_ubyte_file = 'data/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'data/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 'data/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 'data/t10k-labels.idx1-ubyte'

strdel = re.compile(r'[\[\].\s]')



def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        row = images[i].shape[0]
        col = images[i].shape[1]
        for j in range(row):
            for k in range(col):
                if images[i][j][k]>= 127:
                    images[i][j][k] = 1
                else :
                    images[i][j][k] = 0
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def knn(testVec,trainData,trainLabel,k,f,trainNum,testNum):
    trainSize = trainData.shape[0]
    rankList = []
    print(trainSize)
    print(trainData[0].shape[0])
    for i in range(trainSize):
        sum = 0
        for j in range(trainData[i].shape[0]):
            testR = int(testNum[j])
            trainR = int(trainNum[i][j])
            sum += "{0:b}".format(testR^trainR).count('1')
        rankList.append((sum,i))
    rankList.sort(key = lambda x:x[0])
    cnt = []
    if k <= 5 :
        size = k
    else :
        size = 5
    for i in range(size):
        cnt.append(trainLabel[rankList[i][1]])
        print('第%i相似的图像为:' % i,file = f)
        for j in range(trainData[rankList[i][1]].shape[0]):
            print(strdel.sub('',str(trainData[rankList[i][1]][j])),file = f)
    return max(cnt,key = cnt.count)


def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    trainNum = np.empty((train_images.shape[0], train_images[0].shape[0], 1))
    testNum = np.empty((test_images.shape[0], test_images[0].shape[0], 1))
    for i in range(train_images.shape[0]) :
        for j in range(train_images[i].shape[0]) :
            trainNum[i][j] = int(strdel.sub('',str(train_images[i][j])),2)

    for i in range(test_images.shape[0]) :
        for j in range(test_images[i].shape[0]) :
            testNum[i][j] = int(strdel.sub('',str(test_images[i][j])),2)


    print('开始运行')

    # 查看前十个数据及其标签以读取是否正确
    """
    for i in range(10):
        print(train_labels[i])
        print(test_images[i].shape[0:2])
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    """
    acuList = []
    for i in range(1,100,5):
        filename = 'output'+str(i)+'.txt'
        f = open(filename,'wt')
        print('k=%d' % i,file = f)
        rightNum = 0
        wrongNum = 0
        for j in range(test_images.shape[0]):
            print('开始处理第%d个图像:' % j)
            label = knn(test_images[j], train_images, train_labels, i+1,f,trainNum,testNum[j])
            print('第%i幅图像的标签为:%d  正确标签为%d' % (j, label, test_labels[j]))
            print('第%i幅图像的标签为:%d  正确标签为%d' % (j, label, test_labels[j]),file = f)
            if int(label) == int(test_labels[i]):
                rightNum += 1
            else :
                wrongNum += 1
        f.close()
        print(rightNum,wrongNum,(100.0*rightNum/(rightNum+wrongNum)))
        acuList.append((100.0*rightNum/(rightNum+wrongNum),i))
    f = open ('result.txt','wt')
    print(acuList,file = f)
    f.close()



if __name__ == '__main__':
    run()
