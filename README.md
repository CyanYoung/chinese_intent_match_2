## Chinese Intent Match 2018-9

#### 1.preprocess

prepare() 将按类文件保存的数据原地去重，去除停用词，统一替换地区、时间

特殊词，merge() 将数据汇总、打乱，保存为 (text, label) 格式

make_pair() 对每条数据取同类的下一条组合为正例，从其它类数据中

抽样 fold 次组合为反例，汇总、打乱，保存为 (text1, text2, flag) 格式

#### 2.represent

vectorize() 和 vectorize_pair() 分别进行向量化，不处理 label、flag

#### 3.build

train 80% / dev 20% 划分，分别使用 dnn、cnn、rnn 构建匹配模型

#### 4.match

predict() 去除停用词，统一替换地区、时间特殊词，输出相似概率前 3 的语句

#### 5.eval

取相似概率最大语句的标签，test_pair() 和 test() 分别评估匹配、分类的准确率