# Association_Rule
 
#购物车分析

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
