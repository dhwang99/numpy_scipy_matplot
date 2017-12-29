#encoding: utf8

'''
https://ericfu.me/10-minutes-to-pandas/

http://pandas.pydata.org/pandas-docs/stable/10min.html
http://pandas.pydata.org/pandas-docs/stable/cookbook.html

pandas用来做数据分析还是挺有用的，有空好好学习下
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

print '\n#Series是一个值的序列，它只有一列，以及索引. 下例用默认的整数索引'
s = pd.Series([1,3,5, np.nan, 6, 8])

print s


print '\n#DataFrame是有多个列的数据表，每个列拥有一个label, dataFrame也有索引'

dates = pd.date_range('20171201', periods=6)
print dates

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

print df

print '\n#对 DataFrame, 如果参数是一个dict, 则每个dict的v被转为一个Series, 它的k转为label'

df2 = pd.DataFrame({
    'A':1,
    'B':pd.Timestamp('20130102'),
    'C':pd.Series(1, index=list(range(4)), dtype='float32'),
    'D':np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['test', 'train', 'test', 'train']),
    'F': 'foo'
    })

print df2

print '\n#每列的格式用 types 查看'
print df2.dtypes

print '\n#可以认为 DataFrame 就由Series组成的'
df2.A

print '\n#查看数据'

df.head()
df.tail(3)

print '\n# DataFrame内部用numpy格式存储数据，也可以单独查看index, columns'

print df.index

print df.columns

print df.values

print '\n#describe 显示数据摘要'

print df.describe()

print '\n#对数据进行转置'

print df.T

#对axis按照 index排序(axis=0/1)
print df.sort_index(axis=0, ascending=False)
print df.sort_index(axis=1, ascending=False)

print '\n#按值排序'
print df.sort_values(by='B')
print df

print "\n***********数据选择****************"
print "************ 用loc([a:b] 选择时，是包括b的, 如df"
print "以下的方法交互友好，不过作为工业厂景，还是要调用优化后的 .at, .iat, .loc, .iloc, .ix"
print "获取行/列"
print df['A']

"\n通过 label 选择"

df.loc[dates[0]]
#df.loc[dates[0], 1:3]   #需要用iloc
df.loc[dates[0], ['A','B']]
#df.loc[date[0], ['A':'C']]
df.loc[dates[0], 'A':'C']
df.loc['20130102':'20130103', 'A':'D']

df.loc[dates[0]]
df.loc[:, ['A', 'B']]

print "#用loc时，索引数据需要存在，要不然会报错。 如date不能为'20130103'"
#df.loc['20130103', ['A', 'B']]
print df.loc['20171204', ['A', 'B']]

print "#如果所有的维度都写成了标量，这种情况通常用at, 速度更快"
print "#用at时，索引数据需要存在，要不然会报错。 如date不能为'20130103', 此外，要用索引对应的对象，不能用文本"
#df.at['20130103', 'A']  
#df.at['20171203', 'A']  
print df.at[dates[0], 'A']

print "#通过整数下标选数据，和numpy一样"
print df.iloc[1]
print df.iloc[1,2]
print df.iloc[0:2, 1:3]

print "#选多行多列, 坐标可以用list, 也可以用slice"
print df.iloc[3:5, 1:2]
print df.iloc[[1,2,3], [0,2]]
print df.iloc[:, 1:3]
print df.iloc[1,2]

print '#可以用布尔值下标：'
print df[df.A > 0]
print df[df.B > 0]

print "#没有填充的值等于NaN"
print df[df > 0]

print '#isin()函数，是否在集合中'
df2 = df.copy()
df2['E'] = ['one', 'two', 'three', 'four', 'five', 'six']
print df2

isintest = df2['E'].isin(['two', 'four'])
print isintest

print df2[isintest]
print df[isintest]

print "***********设置********************"
print "#为DataFrame增加新列, 按index对应。上面示例已经有了df2['E'] = [...]." 

s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20171202', periods=6))
print s1
df['F'] = s1
print df

print "#通过 label 设置"
df.at[dates[0], 'A'] = 0
df.iat[0,1] = 0
df.loc[:, 'D'] = np.ones(len(df)) * 5
df[df > 0] = -df

print df
print df2

print "***************缺失值***************"
print "pandas 用 np.nan表示缺失值，通常它不会被计算"
print "reindex允许改变某个轴的index, 以下先生成一个示例用的DataFrame:"
df1 = df.reindex(index=dates[0:4], columns = list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
print df1

print "#丢弃有NaN的行"
df1.dropna()

print "#填充缺失值"
df1.fillna(value=5)

print "#获取布尔值的mask: 哪些值是 NaN"
pd.isnull(df1)

print "统计。统计操作会把NaN 排除在外"
df.mean()
df.mean(1)
df.describe()

print "#Apply 函数"
print "对各列进行 cumsum 计算"
print df.apply(np.cumsum)

print "#取列的最大、最小值"
print df.apply(lambda x:x.max() - x.min())

print "#直方图"
s = pd.Series(np.random.randint(0, 7, size=10))
print s
print s.value_counts()

print "#Series 自带很多字符串处理函数，在str属性中， 下面是一个例子:"
s = pd.Series(['A', 'B', 'C', 'Abba', np.nan, 'CABA', 'dog', 'cat'])
print s.str.lower()

print "**************Merge**************"
print "Concat:"
df3 = pd.DataFrame(np.random.randn(10, 4))
print df3

pieces = [df[3:7], df[:3], df[7:]]
print pd.concat(pieces)

print "Join:"
left = pd.DataFrame({'key':['foo', 'foo'], 'lval':[1,2]})
right = pd.DataFrame({'key':['foo', 'foo'], 'rval':[4,5]})

print left
print right

mr = pd.merge(left, right, on='key')
print mr

print "Append:"
df3 = pd.DataFrame(np.random.randn(8, 4), columns=list("ABCD"))
s = df3.iloc[3]
df4= df3.append(s, ignore_index=True)
print df3
print df4

print "Group by: 支持多列group by, 类似数据库"

df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)
                   })

print df.groupby('A').sum()
print df.groupby(['A', 'B']).sum()

print "Reshape:"
print "Stack 层叠"
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two','one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

df2 = df[:4]
print df2

print "stack()把DataFrame的列 变为一列，就是把列压缩到index里"
stacked = df2.stack()
print stacked
print stacked.index

print "只要是MultiIndex都可以用unstack()恢复出列, 默认是把最近一个index解开"
print stacked.unstack()
print stacked.unstack(0)
print stacked.unstack(1)
print stacked.unstack(2)



print "Pivot Table 旋转"
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})

print df

print "pivot 是把原来的数据(values)作为新表的行(index)、列(columns)"

df2 = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
print df2
df3 = pd.pivot_table(df, values=['D', 'E'], index=['A', 'B'], columns=['C'])
print df3


print "时间序列"
print "pandas的时间序列功能在金融应用中很有用"
print "resample功能:"

rng = pd.date_range('20120101', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
rs = ts.resample('T').sum()
print rs

print "时区表示:"
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randint(0, 5, len(rng)), index=rng)

print ts

ts_utc = ts.tz_localize('UTC')
print ts_utc

print "转换时区"
ts2 = ts_utc.tz_convert('US/Eastern')
print ts2

print "ww Timestamp index转换成 TimePeriod"

rng = pd.date_range('20120101', periods=5, freq='M')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)

print ts

ps = ts.to_period()
print ps

print ps.to_timestamp()

print "**************类别************"
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df['grade'] = df['raw_grade'].astype('category')
print df['grade']

print "类别可以 inplace 地赋值：（只是改一下对应的字符串嘛，类别是用 Index 对象存储的）"
df['grade'].cat.categories=["very good", "good", "very bad"]
print df['grade']

print "修改类别时，如果有新的类别，会自动加进去"
df['grade'].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
print df['grade']

print df.groupby('grade').size()


print '对于 DataFrame，可以直接 plot'
rng = pd.date_range('20120101', periods=1000, freq='d')
df = pd.DataFrame(np.random.randn(1000, 4), index=rng, columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')
plt.savefig('pandas_test.png')


print '**********读取，写入数据****************'
print "可以读写csv, hdf, excel格式的文件"

df.to_csv('foo.csv')
df2 = pd.read_csv('foo.csv')

df.to_hdf('foo.h5','df')
df3 = pd.read_hdf('foo.h5','df')

df.to_excel('foo.xlsx', sheet_name='Sheet1')
df4 = pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
