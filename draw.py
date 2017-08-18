import numpy as np
import matplotlib.pyplot as plt


def getInfo(filename):
    loss_list = []
    acc_list = []
    count = 0
    file = open(filename)
    flag1 = True
    flag2 = True
    N = ''
    while 1:
        line = file.readline()
        if not line:
            break
        # find Epoch
        if line[0:5] == 'Epoch' and flag1:
            index = line.find('/')
            Epoch = line[index + 1:]
            flag1 = False
        # find number of x_train
        if flag1 == False and flag2:
            if line.find('/') > 0 and line.find('[') > 0:
                index = line.find('/')
                N = line[index + 1: index + 6]
                flag2 = False

        if line[0:5] == N:
            index = line.find("loss:")
            loss = line[index + 6: index + 12]
            loss_list.append(loss)
            index = line.find("acc:")
            acc = line[index + 5: index + 11]
            acc_list.append(acc)

    return Epoch, loss_list, acc_list


if __name__ == '__main__':
    filename = "1.out"
    Epoch, loss_list, acc_list = getInfo(filename)
    index = np.arange(int(Epoch), dtype=np.int32)
    plt.figure(1)
    plt.plot(index, loss_list)
    plt.savefig('loss.png')
    plt.figure(2)
    plt.plot(index, acc_list)
    plt.savefig('acc.png')
    plt.show()
