# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:42:17 2020

@author: lenovo
"""

import os
os.chdir("C:\\Users\\lenovo\\Desktop\\")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei',style='darkgrid') 

df=pd.read_csv('LCIS.csv')

#统一变量名
df.rename(columns={'ListingId':'id',
                   'recorddate':'记录日期'},inplace=True)
#重复值的处理
df.drop_duplicates(inplace=True)

#缺失值,统计各个字段确实率
missrate=pd.DataFrame(df.apply(lambda x: sum (x.isnull())/len(x)))
missrate.columns=['缺失率']
missrate=missrate[missrate['缺失率']>0]['缺失率'].apply(lambda x: format(x,'.3%'))

#异常值
df1=df[df['下次计划还款利息'].isnull()]['标当前状态'].value_counts()
df.dropna(subset=['记录日期'],inplace=True)
df=df[(df['手机认证']=='成功认证')|(df['手机认证']=='未成功认证')]

##############################贷款前#################################
#借款人年龄分布
'''plt.style.use('ggplot')
plt.boxplot(x =df['年龄'], 
            patch_artist = True, 
            showmeans = True, 
            boxprops = {'color': 'black', 'facecolor': '#9999ff'}, 
            flierprops = {'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'}, 
            meanprops = {'marker': 'D', 'markerfacecolor': 'indianred'}, 
            medianprops = {'linestyle': '--', 'color': 'orange'})
plt.title('借款人年龄分布')
plt.show()'''
#借款人性别分布
'''sns.countplot(data=df,x='性别',palette='Accent',saturation=5)
plt.title('借款人性别分布')'''
#借款类型
'''df_type0 = pd.DataFrame(df.groupby('借款类型')['记录日期'].count())
df_type0['借款类型占比%'] = df_type0['记录日期']/df_type0['记录日期'].sum()
plt.pie(df_type0['记录日期'],labels=df_type0.index,autopct='%.2f%%')
plt.title('借款类型分布')'''
#随年龄借款金额的变化
'''dfrs=pd.DataFrame(df.groupby(['年龄','性别'])['记录日期'].count()).reset_index()
sns.lineplot(data=dfrs,hue='性别',x='年龄',y='记录日期')
plt.title('随年龄借款人数的变化')
plt.ylabel('借款人数')
labels = ['18-23','24-29','30-35','36-41','42-47','48-53','54-59','60-65']
bin_age = [18,24,30,36,42,48,54,60,66]
df['年龄段'] = pd.cut(df['年龄'],bins=bin_age,labels=labels,right=False)
sns.lineplot(data=df[df['借款金额']<10000],hue='性别',x='年龄段',y='借款金额')
plt.title('随年龄段借款金额的变化')'''
#是否首标
'''dfsb = pd.DataFrame(df.groupby('是否首标')['记录日期'].count())
dfsb['借款类型占比%'] = dfsb ['记录日期']/dfsb ['记录日期'].sum()
plt.pie(dfsb ['记录日期'],labels=dfsb.index,autopct='%.2f%%')
plt.title('是否首标借款类型分布')'''
#新老哦用户的金额需求
'''df.loc[df['是否首标']=='是','新老用户'] = '新用户'
df.loc[df['是否首标']!='是','新老用户'] = '老用户'

sns.boxplot(data=df[df['借款金额']<10000],y='借款金额',x='新老用户')
plt.title('新老用户借款金额需求')'''
#借款金额
plt.style.use('ggplot')
plt.boxplot(x =df[df['借款金额']<12000]['借款金额'], 
            patch_artist = True, 
            showmeans = True, 
            boxprops = {'color': 'black', 'facecolor': '#9999ff'}, 
            flierprops = {'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'}, 
            meanprops = {'marker': 'D', 'markerfacecolor': 'indianred'}, 
            medianprops = {'linestyle': '--', 'color': 'orange'})
plt.title('借款金额分布')
plt.xlabel('')
plt.show()
#借款利率
'''plt.style.use('ggplot')
plt.boxplot(x =df['借款利率'], 
            patch_artist = True, 
            showmeans = True, 
            boxprops = {'color': 'black', 'facecolor': '#9999ff'}, 
            flierprops = {'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'}, 
            meanprops = {'marker': 'D', 'markerfacecolor': 'indianred'}, 
            medianprops = {'linestyle': '--', 'color': 'orange'})
plt.title('借款利率分布')
plt.show()'''
#借款期限
'''plt.style.use('ggplot')
plt.boxplot(x =df['借款期限'], 
            patch_artist = True, 
            showmeans = True, 
            boxprops = {'color': 'black', 'facecolor': '#9999ff'}, 
            flierprops = {'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'}, 
            meanprops = {'marker': 'D', 'markerfacecolor': 'indianred'}, 
            medianprops = {'linestyle': '--', 'color': 'orange'})
plt.title('借款期限分布')
plt.show()
'''









###################################################用户基本信息维度
#############性别
df.loc[df['标当前状态']=='逾期中','是否逾期'] = '逾期中'
df.loc[df['标当前状态']!='逾期中','是否逾期'] = '未逾期'
df_sex = df.groupby('性别')['是否逾期'].value_counts()
df_sex = pd.DataFrame(df_sex.unstack(level=1))
df_sex['逾期占比'] = df_sex['逾期中']/(df_sex['逾期中'] + df_sex['未逾期'])
df_sex['总人数'] = df_sex['逾期中'] + df_sex['未逾期']

'''fig=plt.figure(figsize=(12,8))
layout=(1,2)
ax1=plt.subplot2grid(layout,(0,0))
ax2=plt.subplot2grid(layout,(0,1))
ax1.pie(df_sex['总人数'],labels=['女','男'],autopct='%.2f%%')
df_sex['逾期占比'].plot(kind='bar',rot=1,alpha=0.6,ax=ax2)
ax1.set_title('性别贷款占比',fontsize = 18)
ax2.set_title('性别逾期占比',fontsize = 18)'''


##############年龄
'''
sns.distplot(df['年龄'],label='整体年龄分布',hist=False)
sns.distplot(df[df['是否逾期'] =='逾期中']['年龄'],label='逾期人员年龄分布',color='r',hist=False)
sns.distplot(df[(df['是否逾期'] =='逾期中')&(df['性别'] =='女')]['年龄'],label='女性逾期人员年龄分布',color='y',hist=False)
sns.distplot(df[(df['是否逾期'] =='逾期中')&(df['性别'] =='男')]['年龄'],label='男性逾期人员年龄分布',color='g',hist=False)
plt.legend()'''


labels = ['18-23','24-29','30-35','36-41','42-47','48-53','54-59','60-65']
bin_age = [18,24,30,36,42,48,54,60,66]
df['年龄段'] = pd.cut(df['年龄'],bins=bin_age,labels=labels,right=False)
df_age = df.groupby(['年龄段','性别'])['是否逾期'].value_counts()
df_age = df_age.unstack(level=2)
df_age['逾期占比%'] = round(df_age['逾期中']/(df_age['逾期中']+df_age['未逾期'])*100,2)

'''fig=plt.figure(figsize=(10,8))
df_age=df_age.reset_index()
sns.barplot(x='年龄段',y='逾期占比%',hue='性别',data=df_age,palette='Paired')
y = df_age[df_age['性别']=='女']['逾期占比%']
x = len(y)
for i,j in zip(range(x),y):    
    plt.text(i-0.4,j+0.1,'%.2f%%'%j)
y1 = df_age[df_age['性别']=='男']['逾期占比%']
x1 = len(y1)
for i,j in zip(range(x1),y1):
    plt.text(i,j+0.1,'%.2f%%'%j)
plt.title('年龄段逾期占比')'''
    
dfnan=df[(df['年龄段'] =='54-59')&(df['是否逾期']=='逾期中')&(df['性别']=='男')]


'''plt.style.use('ggplot')
plt.boxplot(x =dfnan['借款金额'], 
            patch_artist = True, 
            showmeans = True, 
            boxprops = {'color': 'black', 'facecolor': '#9999ff'}, 
            flierprops = {'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'}, 
            meanprops = {'marker': 'D', 'markerfacecolor': 'indianred'}, 
            medianprops = {'linestyle': '--', 'color': 'orange'})
plt.title('男性54-59岁逾期中的借款金额')
plt.show()'''

'''sns.countplot(data=dfnan,x=dfnan['借款期限'],palette='Accent',saturation=3)
plt.title('男性54-59岁逾期中的借款期限')

sns.countplot(data=dfnan,x=dfnan['初始评级'],palette='Accent',saturation=3)
plt.title('男性54-59岁逾期中的初始评级')

sns.countplot(data=dfnan,x=dfnan['借款利率'],palette='Accent',saturation=2.5)
plt.title('男性54-59岁逾期中的借款利率')'''

###################初始评级
'''df_pre = df.groupby(['性别','初始评级'])['是否逾期'].value_counts()
df_pre = df_pre.unstack(level=2)
df_pre['逾期占比%'] = round(df_pre['逾期中'] / (df_pre['逾期中'] + df_pre['未逾期'])*100,2)

fig=plt.figure(figsize=(10,8))
df_pre=df_pre.reset_index()
sns.barplot(x='初始评级',y='逾期占比%',hue='性别',data=df_pre,palette='Paired')
y = df_pre[df_pre['性别']=='女']['逾期占比%']
x = len(y)
for i,j in zip(range(x),y):    
    plt.text(i-0.4,j+0.001,'%.2f%%'%j)
y1 = df_pre[df_pre['性别']=='男']['逾期占比%']
x1 = len(y1)
for i,j in zip(range(x1),y1):
    plt.text(i,j+0.001,'%.2f%%'%j)
plt.title('初始评级逾期占比')'''

#这里分别观察男性女性的初始评级为e的特点
dfe0=df[(df['初始评级']=='E')&(df['是否逾期']=='逾期中')&(df['性别']=='女')]
dfe1=df[(df['初始评级']=='E')&(df['是否逾期']=='逾期中')&(df['性别']=='男')]
dfe1=dfe1[dfe1['借款金额']<55000]

'''plt.style.use('ggplot')
plt.boxplot(x =dfe0['借款金额'], 
            patch_artist = True, 
            showmeans = True, 
            boxprops = {'color': 'black', 'facecolor': '#9999ff'}, 
            flierprops = {'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'}, 
            meanprops = {'marker': 'D', 'markerfacecolor': 'indianred'}, 
            medianprops = {'linestyle': '--', 'color': 'orange'})
plt.title('女性初始评级为E在逾期中的借款金额')
plt.show()'''

'''plt.style.use('ggplot')
plt.boxplot(x =dfe1['借款金额'], 
            patch_artist = True, 
            showmeans = True, 
            boxprops = {'color': 'black', 'facecolor': '#9999ff'}, 
            flierprops = {'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'}, 
            meanprops = {'marker': 'D', 'markerfacecolor': 'indianred'}, 
            medianprops = {'linestyle': '--', 'color': 'orange'})
plt.title('男性初始评级为E在逾期中的借款金额')
plt.show()'''

'''g = sns.FacetGrid(df, col='初始评级',row = '性别',palette='seismic', size=4)
g.map(sns.countplot, '年龄段', alpha=0.8)
g.add_legend()'''

########################################################用户行为
###################借款类型
'''df_type = df.groupby(['性别','借款类型'])['是否逾期'].value_counts()
df_type = df_type.unstack(level=2)
df_type['逾期占比'] = df_type['逾期中']/(df_type['逾期中']+df_type['未逾期'])
df_type['逾期占比%'] = df_type['逾期中']/(df_type['逾期中']+df_type['未逾期'])*100
y = df_type.loc['女',:]['逾期占比%'].sort_values(ascending=False)
sns.barplot(y.index,y,palette='Accent',saturation=5)
x = len(y)
for i,j in zip(range(x),y):    
    plt.text(i-0.2,j+0.001,'%.2f%%'%j)
plt.title('女性借款类型逾期占比')
plt.legend()


df_type = df.groupby(['性别','借款类型'])['是否逾期'].value_counts()
df_type = df_type.unstack(level=2)
df_type['逾期占比'] = df_type['逾期中']/(df_type['逾期中']+df_type['未逾期'])
df_type['逾期占比%'] = df_type['逾期中']/(df_type['逾期中']+df_type['未逾期'])*100
y = df_type.loc['男',:]['逾期占比%'].sort_values(ascending=False)
sns.barplot(y.index,y,palette='Accent',saturation=5)
x = len(y)
for i,j in zip(range(x),y):    
    plt.text(i-0.2,j+0.001,'%.2f%%'%j)
plt.title('男性借款类型逾期占比')
plt.legend()'''
#######################借款期限
'''df_time = df.groupby(['性别','借款期限'])['是否逾期'].value_counts()
df_time = df_time.unstack(level=2)
df_time['逾期占比'] = df_time['逾期中']/(df_time['逾期中']+df_time['未逾期'])
df_time['逾期占比%'] = df_time['逾期中']/(df_time['逾期中']+df_time['未逾期'])*100
y = df_time.loc['男',:]['逾期占比%'].sort_values(ascending=False)
plt.figure(figsize=(10,5))
y.plot(kind='bar',alpha=0.6,rot=1)
x = len(y)
for i,j in zip(range(x),y):    
    plt.text(i-0.3,j+0.001,'%.2f%%'%j)
plt.title('男性借款日期逾期占比')
plt.legend()'''


'''df_time = df.groupby(['性别','借款期限'])['是否逾期'].value_counts()
df_time = df_time.unstack(level=2)
df_time['逾期占比'] = df_time['逾期中']/(df_time['逾期中']+df_time['未逾期'])
df_time['逾期占比%'] = df_time['逾期中']/(df_time['逾期中']+df_time['未逾期'])*100
y = df_time.loc['女',:]['逾期占比%'].sort_values(ascending=False)
plt.figure(figsize=(10,5))
y.plot(kind='bar',alpha=0.6,rot=1)
x = len(y)
for i,j in zip(range(x),y):    
    plt.text(i-0.3,j+0.001,'%.2f%%'%j)
plt.title('女性借款日期逾期占比')
plt.legend()'''

df24=df[(df['借款期限']==24)&(df['是否逾期']=='逾期中')]

'''sns.countplot(data=df24,x='年龄段',palette='Accent',saturation=5)
plt.title('借款期限为24的年龄段逾期占比')


plt.style.use('ggplot')
plt.boxplot(x =df24['借款金额'], 
            patch_artist = True, 
            showmeans = True, 
            boxprops = {'color': 'black', 'facecolor': '#9999ff'}, 
            flierprops = {'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'}, 
            meanprops = {'marker': 'D', 'markerfacecolor': 'indianred'}, 
            medianprops = {'linestyle': '--', 'color': 'orange'})
plt.title('借款期限为24在逾期中的借款金额')
plt.show()
'''

##############################借款金额
'''df['借款金额区间'] = pd.qcut(df['借款金额'],4)
df_money = df.groupby(['性别','借款金额区间'])['是否逾期'].value_counts()
df_money = df_money.unstack(level=2)
df_money['逾期占比'] = df_money['逾期中']/(df_money['逾期中']+df_money['未逾期'])
df_money['逾期占比%'] = df_money['逾期中']/(df_money['逾期中']+df_money['未逾期'])*100

fig=plt.figure(figsize=(10,8))
df_money=df_money.reset_index()
sns.barplot(x='借款金额区间',y='逾期占比%',hue='性别',data=df_money,palette='Paired')
y = df_money[df_money['性别']=='女']['逾期占比%']
x = len(y)
for i,j in zip(range(x),y):    
    plt.text(i-0.3,j+0.001,'%.2f%%'%j)
y1 = df_money[df_money['性别']=='男']['逾期占比%']
x1 = len(y1)
for i,j in zip(range(x1),y1):
    plt.text(i+0.1,j+0.001,'%.2f%%'%j)
plt.title('借款金额区间逾期占比')
'''




