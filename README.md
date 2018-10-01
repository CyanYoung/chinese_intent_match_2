## Chinese Intent Match 2018-9

#### 1.preprocess

prepare() 将按类文件保存的数据原地去重，去除停用词，统一替换地区、时间等

特殊词，merge() 将数据汇总、打乱，保存为 (text, label) 格式

make_pair() 对每条数据取同类组合为正例，从异类数据中抽样 fold 次

组合为反例，汇总、打乱，保存为 (text1, text2, flag) 格式

flag 代表 proximity，同类为 1、异类为 0，pred 只限于 [0, 1] 区间

#### 2.represent

vectorize() 和 vectorize_pair() 分别进行向量化，不处理 label、flag

#### 3.build

train 80% / dev 20% 划分，分别通过 dnn、cnn、rnn 构建匹配模型

#### 4.encode

定义模型的编码部分、按层名载入相应权重，对训练数据进行预编码

#### 5.match

定义模型的匹配部分、按层名载入相应权重，predict() 读取缓存数据

去除停用词，统一替换地区、时间等特殊词，输出相似概率前 3 的语句

#### 6.eval

取相似概率最大语句的标签，test_pair() 和 test() 分别评估匹配、分类的准确率