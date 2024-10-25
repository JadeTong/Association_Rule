'''
购物车分析
1、数据及分析对象
数据内容来自“在线零售数据集”。该数据集记录了在2010年12月01日至2011年12月09日的541909条在线交易记录，包含8个属性，主要属性如下：
InvoiceNo: 订单编号，由6位整数表示，退货单号由字母“C”开头。
StockCode: 产品编号，每个不同的产品由不重复的5位整数表示。
Description: 产品描述。
Quantity: 产品数量，每笔交易的每件产品的数量。
InvoiceDate: 订单日期和时间，表示生成每笔交易的日期和时间。
UnitPrice: 单价，单位产品的英镑价格。
CustomerID:顾客编号，每个客户由唯一的5位整数表示。
Country: 国家名称，每个客户所在国家/地区的名称。

2、目的及分析任务
计算最小支持度为0.07的德国客户购买产品的频繁项集；
计算最小置信度为0.8且提升度不小于2的德国客户购买产品的关联关系。

3、实现方法及工具
本例采用的是mlxtend包

'''    
#%%                                1.业务理解
# =============================================================================
# 计算德国客户购买产品的频繁项集和关联关系，从而判断哪些产品更可能被同时购买。
# 该业务的主要内容是将最小支持度设为0.07，生成德国客户购买产品的频繁项集，并计算德国客户购买产品中具有正相关关系的商品，
# 并将最小置信度设为0.8，筛选出满嘴最小置信度且提升度不小于2的德国客户购买产品的关联关系。
# =============================================================================
#%%                                2.数据读取
import pandas as pd
retail = pd.read_excel('D:/desktop/ML/关联规则/Online Retail.xlsx')
retail.head()
#%%                                3.数据理解 
#%%%查看数据形状
retail.shape    #(541909,8)
#%%% 查看列名
retail.columns
# =============================================================================
# Index(['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
#        'UnitPrice', 'CustomerID', 'Country'],
#       dtype='object')
# =============================================================================

#%%% 对数据进行探索性分析
retail.describe()
# =============================================================================
#             Quantity  ...     CustomerID
# count  541909.000000  ...  406829.000000
# mean        9.552250  ...   15287.690570
# min    -80995.000000  ...   12346.000000
# 25%         1.000000  ...   13953.000000
# 50%         3.000000  ...   15152.000000
# 75%        10.000000  ...   16791.000000
# max     80995.000000  ...   18287.000000
# std       218.081158  ...    1713.600303
# 
# [8 rows x 4 columns]
# =============================================================================

#%%% 查看各国家的购物数量,即有多少条购买记录
retail.Country.value_counts()
# =============================================================================
# Country
# United Kingdom          495478
# Germany                   9495
# France                    8557
# ......
# Czech Republic              30
# Bahrain                     19
# Saudi Arabia                10
# Name: count, dtype: int64
# =============================================================================

#英国的客户购买商品数量最多，其次为德国，有9495条记录。

#%%% 查看订单编号（InvoiceNo）一列中是否有重复的值
retail.duplicated(subset=['InvoiceNo']).any()    #True
#订单编号有重复表示同一个订单中有多个同时购买的商品，符合apriori算法的数据要求

#%%                                4.数据预处理
#%%查看数据是否有缺失值
retail.isna().sum()
# =============================================================================
# InvoiceNo           0
# StockCode           0
# Description      1454      *******
# Quantity            0
# InvoiceDate         0
# UnitPrice           0
# CustomerID     135080      *******
# Country             0
# dtype: int64
# =============================================================================
#Description和CustomerID有缺失值

#%%% 将商品名称（Description）的字符串头尾的空白字符删除
retail.Description = retail.Description.str.strip()

#%%% 查看Description的缺失值个数
retail.Description.isna().sum()
#1455，比去除空白字符后，缺失值增加了一个

#%%% 去除Description里有缺失值的记录
#dropna(axis=0/1或'index/columns',how='any/all',thresh=None/数字,subset='None/某行/列',inplace=False/True)
retail.dropna(axis=0,how='any',subset='Description',inplace=True)

retail.shape     #(540454,8)

#%%% 由于退货的订单编号有字母C开头，删除含字母C的已取消订单
retail.InvoiceNo = retail.InvoiceNo.astype('str') 
retail_return = retail[retail.InvoiceNo.str.contains('C')]    #订单号带C的有9288个记录
print(retail_return)
#将上面的9288条记录删除
retail = retail[~retail.InvoiceNo.str.contains('C')]

#%%% 考虑到内存限制，本案例只计算德国客户购买的商品的频繁项集及关联规则，全部计算则计算量太大
Germany = retail[retail.Country == 'Germany']
print(Germany)

#%%% 建一个数据透视表，用pivot_table()更高效
shopping_cart = Germany.pivot_table(index='InvoiceNo', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)
shopping_cart = shopping_cart.applymap(lambda x: 1 if x > 0 else 0)
print(shopping_cart)

#%%                              5.生成频繁项集
from mlxtend.frequent_patterns import apriori, association_rules
#%%% 使用 Apriori 生成频繁项集
shopping_cart_bool = shopping_cart.astype(bool)
frequent_itemsets = apriori(shopping_cart_bool, min_support=0.07, use_colnames=True)
print(frequent_itemsets.head())
frequent_itemsets.shape     ##(39,2)
##满足最小支持度0.07的频繁项集有39个

#%%                              6.计算关联度
#将提升度作为度量计算关联规则，并设置阈值为1，表示计算具有正相关关系的关联规则。
# 生成关联规则
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules.head())
# =============================================================================
# 从结果可以看出各项关联规则的详细信息。以第一条关联规则{POSTAGE}————{16 RIBBONS RUSTIC CHARM}为例，
# {POSTAGE}的支持度为0.818381，{16 RIBBONS RUSTIC CHARM}的支持度 0.102845，
# 项集{POSTAGE, 16 RIBBONS RUSTIC CHARM}的支持度为0.091904，
# 客户购买 POSTAGE 时也购买 6 RIBBONS RUSTIC CHARM 的置信度为 0.112299，提升度为1.091933，
# 规则杠杆率leverage为0.007738（即当POSTAGE 和 6 RIBBONS RUSTIC CHARM 独立分布时，二者一起出现的次数比预期多），
# 规则确信度conviction为1.010651（与提升度类似，但用差值表示，确信度值越大则POSTAGE和6 RIBBONS RUSTIC CHARM关联关系越强）。
# =============================================================================

#%%% 查看rules的形状
rules.shape    #(34,10)，总共输出34条关联规则

#%% 接着筛选出提升度不小于2且置信度不小于0.8的关联规则
rules_lift_confi=rules[(rules.lift >= 2) & (rules.confidence >= 0.8)]
rules_lift_confi
# =============================================================================
#                                      antecedents                           consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction  zhangs_metric
# 25           (ROUND SNACK BOXES SET OF 4 FRUITS)  (ROUND SNACK BOXES SET OF4 WOODLAND)            0.157549            0.245077  0.131291    0.833333  3.400298  0.092679     4.52954       0.837922
# 29  (POSTAGE, ROUND SNACK BOXES SET OF 4 FRUITS)  (ROUND SNACK BOXES SET OF4 WOODLAND)            0.150985            0.245077  0.124726    0.826087  3.370730  0.087724     4.34081       0.828405

# 由此可知提升度不小于2且满足最小置信度0.8的强关联规则有两条，{ROUND SNACK BOXES SET OF 4 FRUITS}————>{ROUND SNACK BOXES SET OF4 WOODLAND}
# 和{POSTAGE, ROUND SNACK BOXES SET OF 4 FRUITS}————>{ROUND SNACK BOXES SET OF4 WOODLAND}.
# =============================================================================

#%%                               7.可视化
# 绘制出提升度不小于1的关联规则的散点图，横坐标设置为支持度，纵坐标为置信度，散点的大小表示提升度
import seaborn as sns
import matplotlib.pyplot as plt

# 绘制关联规则的散点图（以支持度和提升度为维度）
plt.figure(figsize=(10, 6))
sns.scatterplot(x='support', y='confidence', size='lift', sizes=(20, 200), hue='lift', data=rules)
plt.title('Support vs Confidence with Lift Size')
plt.show()
















