import tensorflow as tf
import numpy as np

#定义CNN网络实现的类
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	使用embedding，其次是卷积，最大池和softmax层。
    """
#sequence_length: 句子的长度，我们把所有的句子都填充成了相同的长度(该数据集是59)。
#num_classes: 输出层的类别数，我们这个例子是2(正向和负向)。
#vocab_size: 我们词汇表的大小。定义 embedding 层的大小的时候需要这个参数，embedding层的形状是[vocabulary_size, embedding_size]。
#embedding_size: 嵌入的维度。
#filter_sizes: 我们想要 convolutional filters 覆盖的words的个数，对于每个size，我们会有 num_filters 个 filters。比如 [3,4,5] 表示我们有分别滑过3，4，5个 words 的 filters，总共是3 * num_filters 个 filters。
#num_filters: 每一个filter size的filters数量(见上面)。
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):#将train.py中textCNN里定义的参数传进来

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") # input_x输入语料,待训练的内容,维度是sequence_length,"N个词构成的N维向量"
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y") # input_y输入语料,待训练的内容标签,维度是num_classes,"正面 || 负面"
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")# dropout_keep_prob dropout参数,防止过拟合,训练时用

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
		# 指定运算结构的运行位置在cpu非gpu,因为"embedding"无法运行在gpu
        # 通过tf.name_scope指定"embedding"
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
		# filter_sizes卷积核尺寸,枚举后遍历
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]# 4个参数分别为filter_size高h，embedding_size宽w，channel为1，filter个数
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")# W进行高斯初始化
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")# b给初始化为一个常量
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID", # 这里不需要padding
                    name="conv")
                # Apply nonlinearity 激活函数
                # 可以理解为,正面或者负面评价有一些标志词汇,这些词汇概率被增强，即一旦出现这些词汇,倾向性分类进正或负面评价,
                # 该激励函数可加快学习进度，增加稀疏性,因为让确定的事情更确定,噪声的影响就降到了最低。
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
				#池化
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',# 这里不需要padding
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
		#聚合所有池特征
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])# 扁平化数据，跟全连接层相连

        # Add dropout
		# drop层,防止过拟合,参数为dropout_keep_prob
        # 过拟合的本质是采样失真,噪声权重影响了判断，如果采样足够多,足够充分,噪声的影响可以被量化到趋近事实,也就无从过拟合。
        # 即数据越大,drop和正则化就越不需要。
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
		#输出层
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],#前面连扁平化后的池化操作
                initializer=tf.contrib.layers.xavier_initializer())# 定义初始化方式
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			# 损失函数导入
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
			#xw+b
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")#得分函数
            self.predictions = tf.argmax(self.scores, 1, name="predictions")#预测结果

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
		#loss，交叉熵损失函数
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
		#准确率，求和计算算数平均值
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
