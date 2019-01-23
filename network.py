"""
一个实现了随机梯度下降学习算法的前馈神经网络模块。梯度的计算使用到了反向传播。
"""
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """
        在这段代码中，列表 sizes 包含各层神经元的数量。例如，如果我们想创建⼀个在第⼀层有
        2 个神经元，第⼆层有 3 个神经元，最后层有 1 个神经元的 Network 对象，我们应这样写代码：
        net = Network([2, 3, 1])
        Network 对象中的偏置和权重都是被随机初始化的，使⽤ Numpy 的 np.random.randn 函数来⽣
        成均值为 0，标准差为 1 的⾼斯分布。这样的随机初始化给了我们的随机梯度下降算法⼀个起点。
        """
        self.num_layers = len(sizes)# num_layers为层数
        self.sizes = sizes# 列表 sizes 包含各层神经元的数量
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """如果a是输入，则返回网络的输出"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
        """
            weights[0] 30*784矩阵
            a 原始输入 784*1矩阵
            biases[0] 30*1矩阵
            故weights[0]*a为30*1矩阵，weights[0]*a+biases[0]为30*1矩阵
 
            return a
            所以最后结果为10行1列矩阵（向量），对应于0，1，2，3，4，5，6，7，8，9的可能性
        """

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        :param training_data: training_data 是一个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出。
        :param epochs: 变量 epochs为迭代期数量
        :param mini_batch_size: 变量mini_batch_size为采样时的⼩批量数据的⼤⼩
        :param eta: 学习速率
        :param test_data: 如果给出了可选参数 test_data ，那么程序会在每个训练器后评估⽹络，并打印出部分进展。
        这对于追踪进度很有⽤，但相当拖慢执⾏速度。
        :return:
        """
        training_data = list(training_data)#将训练数据集强转为list
        n = len(training_data)#n为训练数据总数，大小等于训练数据集的大小
        if test_data:# 如果有测试数据集
            test_data = list(test_data)# 将测试数据集强转为list
            n_test = len(test_data)# n_test为测试数据总数，大小等于测试数据集的大小
        for j in range(epochs):# 对于每一个迭代期
            random.shuffle(training_data)# shuffle() 方法将序列的所有元素随机排序。

            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            """
                对于下标为0到n-1中的每一个下标，最小数据集为从训练数据集中下标为k到下标为k+⼩批量数据的⼤⼩-1之间的所有元素
                这些最小训练集组成的集合为mini_batches
                mini_batches[0]=training_data[0:0+mini_batch_size]
                mini_batches[1]=training_data[mini_batch_size:mini_batch_size+mini_batch_size]
                ...
            """
            for mini_batch in mini_batches:
                # 对于最小训练集组成的集合mini_batches中的每一个最小训练集mini_batch
                self.update_mini_batch(mini_batch, eta)
                # 调用梯度下降算法
            if test_data:
                # 如果有测试数据集
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test));
                # j为迭代期序号
                # evaluate(test_data)为测试通过的数据个数
                # n_test为测试数据集的大小
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        基于反向传播的简单梯度下降算法更新网络的权重和偏置
        :param mini_batch: 最小训练集
        :param eta: 学习速率
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        """
        运行
        for b in self.biases:
            print("b.shape=",b.shape)
        输出(30，1) (10，1)
        np.zeros((a b))为生成a行b列的数组且每一个元素为0
        所以依次生成一个30行1列的数组和一个10行1列的数组，存放到nabla_b中
        nabla_b[0]为30行1列的数组，每一个元素为0
        nabla_b[1]为10行1列的数组，每一个元素为0
        """

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        """
        同理
        nabla_w[0]为30行784列的数组，每一个元素为0
        nabla_w[1]为10行30列的数组，每一个元素为0
        """
        for x, y in mini_batch:
            # 对于最小训练集中的每一个训练数据x及其正确分类y   x是784*1
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            """
            这行调用了一个称为反向传播的算法，一种快速计算代价函数的梯度的方法。
            delta_nabla_b[0]与biases[0]和nabla_b[0]一样为30*1数组（向量）
            delta_nabla_b[1]与biases[1]和nabla_b[1]一样为10*1数组（向量）
            delta_nabla_w[0]与weights[0]和nabla_w[0]一样为30*784数组（向量）
            delta_nabla_w[1]与weights[1]和nabla_w[1]一样为10*30数组（向量）
            """
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # nabla_b中的每一个即为∂C/∂b
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # nabla_b中的每一个即为∂C/∂w
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        # 更新权重向量组
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        # 更新偏置向量组

    def backprop(self, x, y):
        # 反向传播的算法，一种快速计算代价函数的梯度的方法。
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # 计算 σ函数的导数
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
