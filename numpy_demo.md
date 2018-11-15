
# NumPy

## Array基础属性

### np.array()

- np.array(object:数组对象 , dtype:数组的数据类型 , copy:是否可复制 , order:C(行)、F(列)、A(任意、默认) , subok:默认情况下，返回的数组被强制为基类数组。如果是true则返回子类 , ndmin:指定返回数组的最小维度)


```python
import numpy as np
a = np.array([1, 2, 3])
print(a)
```

    [1 2 3]
    


```python
import numpy as np
#多维度
a = np.array([[1,2],[3,4]])
print(a)
```

    [[1 2]
     [3 4]]
    


```python
import numpy as np
#最小维度
a = np.array([1, 2, 3, 4, 5] , ndmin=2)
print(a)
```

    [[1 2 3 4 5]]
    


```python
import numpy as np
#数组可选数据类型
a = np.array([1, 2, 3, 4] , dtype=complex)
print(a)
```

    [1.+0.j 2.+0.j 3.+0.j 4.+0.j]
    

### 数据类型对象(dtype)

numpy.dtype(object(被转换为数据类型的对象), align(如果为true就向对象添加间隔，使其类似于C的结构体), copy(true，生成dtype对象的新副本，false，结果是内奸数据类型的引用。))


```python
# 使用数组标量类型
import numpy as np
dt = np.dtype(np.int32)
print(dt)
```

    int32
    


```python
# int8 int16 int32 int64 可替换为 i1 i2 i3 i4 , 以及其他
import numpy as np
dt = np.dtype('i4')
print(dt)
```

    int32
    


```python
# 使用端记号
import numpy as np 
dt = np.dtype('>i4')
print(dt)
```

    >i4
    

- 字节顺序取决于数据类型的前缀‘<’或‘>’。‘<’意味着编码是小端(最小有效字节存储在最小地址中)。‘>’意味着编码是大端(最大有效字节存储在最小小地址中)。


```python
# 创建结构化数据类型
import numpy as np
dt = np.dtype([('age', np.int8)])
print(dt)
```

    [('age', 'i1')]
    


```python
# 将以上的数据类型应用于ndarray对象
import numpy as np

dt = np.dtype([('age' , np.int8)])
a = np.array([(10,),(20,),(30,)], dtype=dt)
print(a)
```

    [(10,) (20,) (30,)]
    


```python
# 文件名称可用于访问age列的内容
import numpy as np

dt = np.dtype([('age', np.int8)])
a = np.array([(10,),(20,),(30,)] , dtype=dt)
print(a['age'])
```

    [10 20 30]
    


```python
# 构建Student结构化数据，其中包含字符串字段name，整数字段age和浮点数字段marks
import numpy as np

dt = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(dt)
student = np.array([('zhaozhao', 12, 12.), ('lulu', 11, 11.)])
print(student)
```

    [('name', 'S20'), ('age', 'i1'), ('marks', '<f4')]
    [['zhaozhao' '12' '12.0']
     ['lulu' '11' '11.0']]
    

**每个内建类型都有唯一定义他们的字符代码**
- 'b':布尔类型
- 'i':符号整数
- 'u':无符号整数
- 'f':浮点
- 'c':复数浮点
- 'm':时间间隔
- 'M':日期时间
- 'O':Python对象
- 'S','a':字节串
- 'U':Unicode
- 'V':原始数据

## NumPy-数组属性

### ndarray.shape

- 返回数组维度
- 调整数组大小


```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)
```

    (2, 3)
    


```python
# 调整数组维度
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape = (3 , 2)
print(a)
```

    [[1 2]
     [3 4]
     [5 6]]
    


```python
# 调整数组维度
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]]).reshape(3, 2)
print(a)
```

    [[1 2]
     [3 4]
     [5 6]]
    

### ndarray.ndim

- 返回数组维数


```python
# 创建等间隔数组
import numpy as np
a = np.arange(24)
print(a)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
    


```python
import numpy as np
a = np.arange(24)
print(a.ndim)#1
b = a.reshape(2, 4, 3)
print(b)
#b有三个维度
```

    1
    [[[ 0  1  2]
      [ 3  4  5]
      [ 6  7  8]
      [ 9 10 11]]
    
     [[12 13 14]
      [15 16 17]
      [18 19 20]
      [21 22 23]]]
    

### numpy.itemsize


```python
# dtype为int8（一个字节）
import numpy as np
x = np.array([1,2,3,4,5] , dtype=np.int8)
print(x.itemsize)
```

    1
    


```python
# float32四个字节
import numpy as np
x = np.array([1,2,3,4,5] , dtype=np.float32)
print(x.itemsize)
```

    4
    

### numpy.flags


```python
# 这个字数太多了就不写了值得注意的是WRITEABLE : True这个属性，可设置数据读写属性
import numpy as np
x = np.array([1,2,3,4])
print(x.flags)
```

      C_CONTIGUOUS : True
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False
      UPDATEIFCOPY : False
    

## 数组创建

### numpy.empty
numpy.empty(shape:空数组的形状，整数或整数元组, dtype:数组类型, order:'C'为按行的C风格数组，'F'为按列的数组)
数组的元素都是随机值


```python
import numpy as np
x = np.empty([3,2],dtype=float)
print(x)
```

    [[1.097e-321 0.000e+000]
     [0.000e+000 0.000e+000]
     [0.000e+000 0.000e+000]]
    

### numpy.zeros
参数同上，全0数组


```python
# 元素的数据类型默认为float
import numpy as np
x = np.zeros((3, 4) )
print(x)
```

    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    


```python
# 自定义类型
import numpy as np
x = np.zeros((2, 2), dtype = [('x', 'i4'), ('y', 'i4')])
print(x)
```

    [[(0, 0) (0, 0)]
     [(0, 0) (0, 0)]]
    

### numpy.ones
同上，全1数组

## 来自现有数据的数组

### numpy.asarray
参数同上，将其他对象转换成ndarray类型的对象


```python
# 将列表中转换成ndarray
import numpy as np
x = [1, 2, 3]
a = np.asarray(x)
print(a)
```

    [1 2 3]
    


```python
# 复合类型的列表
import numpy as np
x = [(1, 2, 3), (4, 5), (6, 7, 8)]
a = np.array(x)
print(a)
```

    [(1, 2, 3) (4, 5) (6, 7, 8)]
    

### numpy.frombuffer
将缓冲区解释为一位数组
- buffer:任何暴露缓冲区借口的对象
- dtype:数据类型
- count:读取的数据量，默认是-1，读取所有数据
- offset:需要读取的起始位置，默认是0



```python
import numpy as np
s = b'Hello World'
a = np.frombuffer(s , dtype='S1')
print(a)
```

    [b'H' b'e' b'l' b'l' b'o' b' ' b'W' b'o' b'r' b'l' b'd']
    

### numpy.fromiter
从任何可迭代对象创建一个ndarray对象，返回一个一维数组
- iterable：可迭代对象
- dtype
- count

## 来自数值范围的数组

### numpy.arange
等间隔数组
- start,默认是0
- stop，不包含([start, stop) )
- step,步长默认0
- dtype


```python
import numpy as np
x = np.arange(10 , 20 , 5)
print(x)
```

    [10 15]
    

### numpy.linspace
类似于arange，不同的是指定范围内均匀间隔的数量而不是步长
- start
- stop
- num：间隔的数量，默认是50
- endpoint:布尔类型，如果是true，stop的值将包含在序列中
- restep：如果是true，返回样例，以及连续数字之间的步长
- dtype


```python
import numpy as np
x = np.linspace(10 , 20 , 5 , endpoint=False)
print(x)
x = np.linspace(10 , 20 , 5 , endpoint=True)
print(x)
```

    [10. 12. 14. 16. 18.]
    [10.  12.5 15.  17.5 20. ]
    

## numpy.logspace
对数刻度上均匀分布的数字，底数通常为10
- start
- stop
- num
- endpoint
- base 对数空间的底数，默认是10
- dtype


```python
import numpy as np
a = np.logspace(1,10,num = 10 , base = 2)
print(a)
```

    [   2.    4.    8.   16.   32.   64.  128.  256.  512. 1024.]
    

## 切片和索引

### 使用slice
- start
- stop
- step


```python
import numpy as np
a = np.arange(10)
s = slice(2, 7, 2)
print(a[s])
```

    [2 4 6]
    

### 使用：
start:stop:step


```python
import numpy as np
a = np.arange(10)
b = a[2:7:2]
print(b)
```

    [2 4 6]
    

### 使用...


```python
import numpy as np
a = np.arange(16).reshape(4,4)
print('原数组')
print(a)
print('第二列')
print(a[...,2])
print('第二列往后')
print(a[...,2:])
print('第二行')
print(a[2,...])
```

    原数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    第二列
    [ 2  6 10 14]
    第二列往后
    [[ 2  3]
     [ 6  7]
     [10 11]
     [14 15]]
    第二行
    [ 8  9 10 11]
    

## 高级索引
### 整数索引


```python
import numpy as np
x = np.arange(12).reshape(3,4)
y = x[[0,1,2],[0,0,0]]#相当于下标(0,0) (1,0) (2,0)
print(y)
```

    [0 4 8]
    

- 也可以使用切片

### 布尔索引

当结果对象是布尔运算的的结果时，将使用此类型的高级索引

**比较**


```python
import numpy as np 
x = np.arange(12).reshape(3,4)
print(x[x>5])
```

    [ 6  7  8  9 10 11]
    

**isnan**


```python
# 使用~取补运算符来过滤NaN
import numpy as np
a = np.array([[1, np.nan , 2] ,[ 3,2, np.nan],[3, 4, 5] ])
print(a[~np.isnan(a)])
```

    [1. 2. 3. 2. 3. 4. 5.]
    

#### iscomplex


```python
import numpy as np
a = np.array([[1, np.nan , 2.7+3j] ,[ 3,2, np.nan],[3, 3.5+5j, 5] ])
print(a[np.iscomplex(a)])
```

    [2.7+3.j 3.5+5.j]
    

## 广播


```python
# 基本的广播
import numpy as np
a = np.arange(12).reshape(3,4)
b = np.arange(12).reshape(3,4)
print(a,"\n\n",b,'\n\n',a+b)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]] 
    
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]] 
    
     [[ 0  2  4  6]
     [ 8 10 12 14]
     [16 18 20 22]]
    


```python
# 小数组会广播到大数数组中
import numpy as np
a = np.arange(12).reshape(3,4)
b = np.arange(4)
print(a,"\n\n",b,'\n\n',a+b)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]] 
    
     [0 1 2 3] 
    
     [[ 0  2  4  6]
     [ 4  6  8 10]
     [ 8 10 12 14]]
    

## 数组上的迭代

### numpy.nditer


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原始数组')
print(a)
print('迭代后')
for x in np.nditer(a):
    print(x)
print('转置后')
print(a.T)
print('转置后迭代')
for x in np.nditer(a.T):
    print(x)
```

    原始数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    迭代后
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    转置后
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]
    转置后迭代
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    

### 迭代顺序
如果相同元素使用F风格顺序存储，则迭代器选择以更有效的方式对数组进行迭代


```python
import numpy as np
a = np.arange(12).reshape(3,4)
import numpy as np
a = np.arange(12).reshape(3,4)
print('原始数组')
print(a)
print('迭代后')
for x in np.nditer(a):
    print(x)
print('转置后')
print(a.T)
print('转置后迭代')
for x in np.nditer(a.T):
    print(x)
b = a.T.copy(order='C')
print('C原始数组')
print(b)
for x in np.nditer(b):
    print(x)
c = a.T.copy(order="F")
print('F原始数组')
print(c)
for x in np.nditer(c):
    print(x)

print('CF强制')
print('C风格排序')
for x in np.nditer(a.T , order='C'):
    print(x)
print('F风格排序')
for x in np.nditer(a.T , order='F'):
    print(x)
```

    原始数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    迭代后
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    转置后
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]
    转置后迭代
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    C原始数组
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]
    0
    4
    8
    1
    5
    9
    2
    6
    10
    3
    7
    11
    F原始数组
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    CF强制
    C风格排序
    0
    4
    8
    1
    5
    9
    2
    6
    10
    3
    7
    11
    F风格排序
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    

### 修改数组的值

nditer对象的另一个可选参数op_flags。其默认值是制度，但是可以设置为读写模式或者只写模式，这将允许更改数组


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原始数组')
print(a)
for x in np.nditer(a , op_flags = ['readwrite']):
    x[...]=2*x
print('修改后的数组')
print(a)
```

    原始数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    修改后的数组
    [[ 0  2  4  6]
     [ 8 10 12 14]
     [16 18 20 22]]
    

### 外部循环
nditer类的构造器有flags参数，可以接受以下列值
- c_index:可以跟踪C顺序的索引
- f_index:。。。。F。。。。。
- multi-index:每次迭代可以跟踪一种索引值
- external_loop:给出的值是具有多个值的一维数组，而不是零维数组


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原始数组')
print(a)
print('修改后的数组external_loop')
for x in np.nditer(a , flags=['external_loop'], order='F'):
    print(x)
```

    原始数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    修改后的数组external_loop
    [0 4 8]
    [1 5 9]
    [ 2  6 10]
    [ 3  7 11]
    

### 广播迭代

如果两个数组是可广播的，nditer可以同时迭代他们，比如3X4和1X4的数组可以同时被迭代


```python
import numpy as np
a = np.arange(12).reshape(3,4)
b = a[0,...]
print('a原始数组')
print(a)
print('b原始数组')
print(b)
print('广播迭代之后')
for x,y in np.nditer([a,b]):#注意构建列表
    print('%s:%s'%(x,y))
```

    a原始数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    b原始数组
    [0 1 2 3]
    广播迭代之后
    0:0
    1:1
    2:2
    3:3
    4:0
    5:1
    6:2
    7:3
    8:0
    9:1
    10:2
    11:3
    

## 数组操作

### 修改形状

#### numpy.reshape
- arr:要修改的数组
- newshape:整数或整数数组，新的形状应该兼容原始形状
- order:CFA



```python
#略
```

#### numpy.ndarray.flat
该函数返回数组上的一维迭代器，行为类似于Python的内建迭代器



```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原始数组')
print(a)
print('调用了flat之后')
print(a.flat)
for x in a.flat:
    print(x)
```

    原始数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    调用了flat之后
    <numpy.flatiter object at 0x0000000004D0F940>
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    

#### numpy.ndarray.flatten
返回折叠为一维数组的副本
- order：C:按行 F:按列 A:原顺序 k:元素在内存中出现的顺序


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原始数组')
print(a)
print('调用flatten之后')
print(a.flatten())
print('按行C',a.flatten(order='C'))
print('按列F',a.flatten(order='F'))
print('原顺序A',a.flatten(order='A'))
print('内存顺序k',a.flatten(order='k'))
```

    原始数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    调用flatten之后
    [ 0  1  2  3  4  5  6  7  8  9 10 11]
    按行C [ 0  1  2  3  4  5  6  7  8  9 10 11]
    按列F [ 0  4  8  1  5  9  2  6 10  3  7 11]
    原顺序A [ 0  1  2  3  4  5  6  7  8  9 10 11]
    内存顺序k [ 0  1  2  3  4  5  6  7  8  9 10 11]
    

### 翻转操作

#### numpy.transpose
范栓数组的维度
- arr：要转置的数组
- axes:整数的列表，对应维度，通常所有维度都会反转


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原数组')
print(a)
print('转置数组')
print(np.transpose(a))
```

    原数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    转置数组
    [[ 0  4  8]
     [ 1  5  9]
     [ 2  6 10]
     [ 3  7 11]]
    

#### numpy.ndarray.T
类似于transpose

#### numpy.rollaxis
向后滚动特定的轴，直到一个特定的位置
- arr
- axis:要向后滚动的轴，其他的轴位置不会变
- start:默认为0，表示完整滚动。会滚动到特定位置


```python
import numpy as np
a = np.arange(27).reshape(3,3,3)
print('原数组')
print(a)
print('将轴2滚动到轴0 宽度到深度')
print(np.rollaxis(a, axis=2))
print('将轴0滚动到轴1 宽度到高度')
print(np.rollaxis(a, axis=2,start=0))
```

    原数组
    [[[ 0  1  2]
      [ 3  4  5]
      [ 6  7  8]]
    
     [[ 9 10 11]
      [12 13 14]
      [15 16 17]]
    
     [[18 19 20]
      [21 22 23]
      [24 25 26]]]
    将轴2滚动到轴0 宽度到深度
    [[[ 0  3  6]
      [ 9 12 15]
      [18 21 24]]
    
     [[ 1  4  7]
      [10 13 16]
      [19 22 25]]
    
     [[ 2  5  8]
      [11 14 17]
      [20 23 26]]]
    将轴0滚动到轴1 宽度到高度
    [[[ 0  3  6]
      [ 9 12 15]
      [18 21 24]]
    
     [[ 1  4  7]
      [10 13 16]
      [19 22 25]]
    
     [[ 2  5  8]
      [11 14 17]
      [20 23 26]]]
    

#### numpy.swapaxes
交换数组的两个轴
- arr
- axis1
- axis2

### 修改维度
#### numpy.broadcast
返回一个对象，该对象封装了将一个数组广播到另一个数组的结果
- arr1
- arr2


```python
import numpy as np
x1 = np.arange(12).reshape(3,4)
x2 = x1[0,...]
print('原数组x1')
print(x1)
print('原数组x2')
print(x2)
print('广播之后')
print(np.broadcast(x2,x1))
print(np.broadcast(x2,x1).iters)
y1 , y2 = np.broadcast(x2,x1).iters

```

    原数组x1
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    原数组x2
    [0 1 2 3]
    广播之后
    <numpy.broadcast object at 0x0000000004B2D8A0>
    (<numpy.flatiter object at 0x0000000004C37900>, <numpy.flatiter object at 0x0000000004CBF0A0>)
    

#### numpy.broadcast_to
将数组广播到新的形状。在原始数组上返回只读视图，形状不符合Numpy的广播规则，就会抛出ValueError的错误
- array
- shape
- subok


```python
import numpy as np
x = np.arange(12).reshape(1,12)
print('原数组')
print(x)
print('之后')
print(np.broadcast_to(x , (4,12)))
```

    原数组
    [[ 0  1  2  3  4  5  6  7  8  9 10 11]]
    之后
    [[ 0  1  2  3  4  5  6  7  8  9 10 11]
     [ 0  1  2  3  4  5  6  7  8  9 10 11]
     [ 0  1  2  3  4  5  6  7  8  9 10 11]
     [ 0  1  2  3  4  5  6  7  8  9 10 11]]
    

#### numpy.expand_dims
在指定位置插入新的轴来扩展数组的形状
- arr
- axis:新轴插入的位置



```python
import numpy as np
x = np.array(([1,2],[3,4]))
print('原始数组')
print(x)
y = np.expand_dims(x,axis=0)
print('在位置0插入轴')
print(y)
print(y.shape)
```

    原始数组
    [[1 2]
     [3 4]]
    在位置0插入轴
    [[[1 2]
      [3 4]]]
    (1, 2, 2)
    

#### numpy.squeeze
函数从给定的数组的形状中删除一维条目，此函数需要两个参数
- arr
- axis

### 数组的连接
#### numpy.concatenate
用于沿指定轴连接相同形状的两个或多个数组
- (a1,a2,a3...)
- axis:默认是0


```python
import numpy as np
a = np.arange(12).reshape(3,4)
b = np.arange(12).reshape(3,4)
print('a')
print(a)
print('b')
print(b)
print('沿0轴连接')
print(np.concatenate((a,b)))
print('沿1轴')
print(np.concatenate((a,b),axis = 1))
```

    a
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    b
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    沿0轴连接
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    沿1轴
    [[ 0  1  2  3  0  1  2  3]
     [ 4  5  6  7  4  5  6  7]
     [ 8  9 10 11  8  9 10 11]]
    

#### numpy.stack
沿新轴连接数组序列
- arrays:形状相同的数组序列
- axis:返回数组中的轴，输入数组沿着它来堆叠



```python
import numpy as np
a = np.arange(12).reshape(3,4)
b = np.arange(12).reshape(3,4)
print('a')
print(a)
print('b')
print(b)
print('沿0轴连接')
print(np.stack((a,b)))
print('沿1轴')
print(np.stack((a,b),axis = 1))
```

    a
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    b
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    沿0轴连接
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]]
    沿1轴
    [[[ 0  1  2  3]
      [ 0  1  2  3]]
    
     [[ 4  5  6  7]
      [ 4  5  6  7]]
    
     [[ 8  9 10 11]
      [ 8  9 10 11]]]
    

#### numpy.hstack
通过堆叠来生成水平的单个数组



```python
import numpy as np
a = np.arange(12).reshape(3,4)
b = np.arange(12).reshape(3,4)
print('a')
print(a)
print('b')
print(b)
print('沿1轴连接')
print(np.hstack((a,b)))
```

    a
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    b
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    沿0轴连接
    [[ 0  1  2  3  0  1  2  3]
     [ 4  5  6  7  4  5  6  7]
     [ 8  9 10 11  8  9 10 11]]
    沿1轴
    [[ 0  1  2  3  0  1  2  3]
     [ 4  5  6  7  4  5  6  7]
     [ 8  9 10 11  8  9 10 11]]
    

#### numpy.hstack
通过堆叠来生成竖直的单个数组


```python
import numpy as np
a = np.arange(12).reshape(3,4)
b = np.arange(12).reshape(3,4)
print('a')
print(a)
print('b')
print(b)
print('沿0轴连接')
print(np.vstack((a,b)))
```

    a
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    b
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    沿0轴连接
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    

### 数组的分割
#### numpy.split
沿特定轴将数字分割成子数组
- arr
- indices_or_sections：当传入整数时，表示分割的块数；传入数组时，表示分割的位置
- axis


```python
import numpy as np
a = np.arange(12)
print(a)
print(np.split(a , 3))
print(np.split(a,[3,5,9]))
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11]
    [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
    [array([0, 1, 2]), array([3, 4]), array([5, 6, 7, 8]), array([ 9, 10, 11])]
    

#### numpy.hsplit
将数组按行分割成子数组
#### numpy.vsplit
将数组按列分割成子数组

### 添加/删除元素
#### numpy.resize
- arr
- shape:返回数组的新形状


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原始数组')
print(a)
print('第一个数组')
print(np.resize(a , (4,3)))
print('第二个数组')
print(np.resize(a, (4,4)))
```

    原始数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    第一个数组
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    第二个数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [ 0  1  2  3]]
    

#### numpu.append
再输入的数组末尾添加值，输入数组的维度必须匹配否则会报错ValueError
- arr
- values
- axis

#### numpy.insert
沿定州在输入数组中插入值
- arr
- obj:索引
- values:可以使整数，也可以是数组
- axis:未指定轴将会被展开


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print(np.insert(a , 3 , [111,222]))
print(np.insert(a, 2, 111 , axis=0))
print(np.insert(a, 2, [111,222,333], axis=1))
print(np.insert(a, 2, 111 , axis=1))
```

    [  0   1   2 111 222   3   4   5   6   7   8   9  10  11]
    [[  0   1   2   3]
     [  4   5   6   7]
     [111 111 111 111]
     [  8   9  10  11]]
    [[  0   1 111   2   3]
     [  4   5 222   6   7]
     [  8   9 333  10  11]]
    [[  0   1 111   2   3]
     [  4   5 111   6   7]
     [  8   9 111  10  11]]
    

#### numpy.delete
同insert
- arr
- obj
- axis

#### numpy.unique
返回数组中的去重元素数组
- arr
- return_index:truen返回数组中元素的下标
- return_inverse:true返回去重数组的下标，用于重构输入数组
- return_counts:true返回去重数组中元素在原数组中出现的次数

### 位操作
#### bitwise_and
按位与操作



```python
import numpy as np
print('13和17的二进制形式')
print(bin(13),bin(17))
print('按位与操作')
print(np.bitwise_and(13,17))
```

    13和17的二进制形式
    0b1101 0b10001
    按位与操作
    1
    


#### bitwise_or
按位或操作


#### np.invert
返回补码数字



```python
import numpy as np
print('13的位翻转')
print(np.invert(np.array([13] , dtype=np.uint8)))
print('13的二进制',np.binary_repr(13,width=8))
print('242的二进制',np.binary_repr(242,width=8))
```

    13的位翻转
    [242]
    13的二进制 00001101
    242的二进制 11110010
    

#### np.left_shift
二进制左移


```python
import numpy as np
print('10左移两位', np.left_shift(10,2))
print('10的二进制',np.binary_repr(10,width=8))
print('40的二进制',np.binary_repr(40,width=8))
```

    10左移两位 40
    10的二进制 00001010
    40的二进制 00101000
    

#### np.right_shift
二进制右移

### 字符串函数
#### numpy.char.add
字符串连接


```python
import numpy as np
print(np.char.add(['hello'] , ['kugou']))
print(np.char.add(['a', 'b'],['c', 'd']))
```

    ['hellokugou']
    ['ac' 'bd']
    

#### numpy.char.multiply
多重连接


```python
import numpy as np
print(np.char.multiply('Hello' , 3))
```

    HelloHelloHello
    

#### numpy.char.center
返回所需宽度的数组以便字符串位于中心
- char
- length
- fillchar:左侧右侧进行填充的字符串


```python
import numpy as np
print(np.char.center('NiuB',20,fillchar='*'))
```

    ********NiuB********
    

#### numpy.char.capitalize
返回字符串的副本，第一个字母大写
#### numpy.char.title
返回字符串的副本，每个单词都是首字母大写
#### numpy.char.lower
返回一个数组，所有元素都转换成小写
#### numpy.char.upper
返回一个数组，所有元素都转换成大写
#### numpy.char.split
返回一个数组，默认以空格分开,自定义需要定义sep参数,如sep=','
#### numpy.char.splitlines
返回一个数组，以换行符(\n，\r)分割
#### numpy.char.strip
返回一个数组的副本，移除开头或结尾处特定的字符


```python
import numpy as np
print(np.char.strip('asff dfasdaa' , 'a'))
print(np.char.strip(['asd','sffssaa','aasdaffea'],'a'))
```

    sff dfasd
    ['sd' 'sffss' 'sdaffe']
    

#### numpy.char.join
返回一个数组，由指定的字符连接


```python
import numpy as np
print(np.char.join(':','noibi'))
print(np.char.join([':','-'],['sjisahfo','sjdhfau']))
```

    n:o:i:b:i
    ['s:j:i:s:a:h:f:o' 's-j-d-h-f-a-u']
    

#### numpy.char.replace
```
replace('he is a good boy','is', 'was')
```
#### numpy.char.decode
编码
#### numpy.char.encode
解码
- str
- code=??

## 算数函数
### 三角函数
- np.pi
- np.sin
- np.cos
- np.tan
- np.arcsin
- np.arccos
- np.arctan
- np.degrees

### 舍入函数
#### numpy.around
四舍五入
- arr
- decimals:正数：小数点右侧多少位；负数：小数点左侧多少位
#### numpy.floor
下取整
#### numpy.ceil
上取整
### 算数运算
#### np.add
#### np.subtract
#### np.multipy
#### np.divide
#### np.reciprocal
返回倒数，对于元素大于1 的整数整数整数整数结果始终为0，对于整数0发出溢出警告
#### np.power
求数组的幂
- arr
- arg:指数
#### np.mod ， np.remainder
求余数
#### np.real
实数的实部
#### np.imag
实数的虚部
#### np.conj
共轭复数
#### np.angle
返回复数的角度
- degree:true返回角度，否则返回弧度


### 统计函数
#### np.amin np.amax
返回指定轴上的最小值和最大值的数组
- arr
- axis
#### np.ptp
返回沿轴方向的值范围，即最大值-最小值
- arr
- axis:不指定就是数组中最大值-最小值


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原数组')
print(a)
print('调用ptp')
print(np.ptp(a))
print('沿0轴')
print(np.ptp(a,axis=0))
print('沿1轴')
print(np.ptp(a,axis=1))
```

    原数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    调用ptp
    11
    沿0轴
    [8 8 8 8]
    沿1轴
    [3 3 3]
    

#### np.percentile
百分位数
- arr
- q要计算的百分位数
- axis
#### np.median
中值定义，分开数据上半部分和下半部分的值


```python
import numpy as np
a = np.arange(12).reshape(3,4)
print('原数组')
print(a)
print('调用median')
print(np.median(a))
print('沿0轴')
print(np.median(a,axis=0))
print('沿1轴')
print(np.median(a,axis=1))
```

    原数组
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    调用median
    5.5
    沿0轴
    [4. 5. 6. 7.]
    沿1轴
    [1.5 5.5 9.5]
    

### 排序、搜索、计数函数
#### 排序(快排“quicksort”,归并排序“mergesort”，堆排序“headsort”)
#### np.sort
返回排序之后的副本
- arr
- axis
- kind:排序的方式，默认快排
- order:如果数组包含字段，则是要排序的字段

#### np.argsort
沿给定轴执行间接排序，并返回指定排序类型返回数据的索引数组，这个数组用于构造排序后的数组


```python
import numpy as np
x = np.array([3, 1, 2])
print('原数组')
print(x)
print('调用argsort之后')
y = np.argsort(x)
print(y)
print(x[y])
```

    原数组
    [3 1 2]
    调用argsort之后
    [1 2 0]
    [1 2 3]
    

#### np.lexsort
该函数返回一个索引数组，使用键序列执行间接排序，最后一个键恰好是sort的主键



```python
import numpy as np
key=('a','ab','bc','ba')
values=('sdf','wer','tyu','fghk')
index = np.lexsort((values, key))
print(index)

```

    [0 1 3 2]
    

#### np.argmax np.argmin
返回沿给定轴的最大值和最小值的索引
#### np.nonzero
返回数组中非0元素的索引
#### np.where
返回数组中满足给定条件元素的索引


```python
import numpy as np
x = np.arange(12).reshape(3,4)
print(np.where(x>5))
y = np.where(x>5)
print(x[y])
```

    (array([1, 1, 2, 2, 2, 2], dtype=int64), array([2, 3, 0, 1, 2, 3], dtype=int64))
    [ 6  7  8  9 10 11]
    

#### np.extract
返回满足任何条件的元素



```python
import numpy as np
x = np.arange(12).reshape(3,4)
print(np.extract(x>5,x))
```

    [ 6  7  8  9 10 11]
    

### 矩阵库
#### np.matlib.empty
- shape:定义矩阵的形状可以使整数或者整形的元组
- dtype:可选
- order:CF


```python
import numpy.matlib
import numpy as np
print(np.matlib.empty((2,2)))
```

    [[4.79243676e-322 2.07955588e-312]
     [2.10077583e-312 2.05833592e-312]]
    

#### np.matlib.zeros
#### np.matlib.ones
#### np.matlib.eye
这个函数返回一个矩阵，对角线元素是1，其他位置是0,。
- n返回矩阵的行数
- M返回矩阵的列数，默认为n
- k对角线的索引
- dtype输出数组的类型


```python
import numpy.matlib
import numpy as np
print(np.matlib.eye(n=3,M=4,k=0,dtype=float))
```

    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]]
    

#### np.matlib.identity
返回给定大小的单位单位矩阵，单位矩阵的主对角线元素都是1的方阵


```python
import numpy.matlib
import numpy as np
print(np.matlib.identity(5, dtype=float))
```

    [[1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1.]]
    

#### numpy.matlib.rand
返回给定大小的随即矩阵



```python
import numpy.matlib
import numpy as np
print(np.matlib.rand(3, 3))
```

    [[0.515357   0.26420014 0.59821632]
     [0.10342783 0.99165074 0.90940059]
     [0.35923784 0.39490904 0.03759437]]
    

### 线性代数
[[a,b],[c,d]] [[e,f],[g,h]]
#### np.dot 两个数组的点击
[[a*e+b*g,a*f+b*h],[c*e+d*g,c*f+d*h]]
#### np.vdot 两个向量的点击
a*e+b*f+c*g+d*h
#### np.inner两个数组的内积
返回一维数组的向量内积，对于更高纬度，他返回最后一个轴上的和的乘积
[[a*e+b*f , a*g+b*h],[c*e+d*f , c*g+d*h]]
#### np.matmul两个数组的矩阵积
#### numpy.linalg.det 行列式的值
行列式的计算
#### np.linalg.solve
求线性方程的解
#### np.linalg.inv
计算矩阵的逆



```python
import numpy as np
print(np.inner(np.array([[1,1],[1,1]]) , np.array([[1,1],[1,1]])))
```

    [[2 2]
     [2 2]]
    


```python
import numpy as np
print(np.arange(24).reshape(2,3,4))
```

    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]
    
