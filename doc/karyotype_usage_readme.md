# karyotype 类的试用

## 程序文件
./config/karyotype.ini : 核型图的基本参数配置文件
./utils/chromo_cv_utils.py : 染色体相关图像处理工具类
./utils/utils.py : 核型图相关图像处理工具类
./karyotype_usage_readme.md : 核型图类的使用说明文档
./requirements.txt : 依赖包列表
./karyotype.py : 核型图类
./main.py : 核型图类的使用示例

## 实例初始化

### 核型图的基本参数配置

1. 参数配置文件
  ./config/karyotype.ini ， 该文件中包含了核型图的基本参数配置，已经对当前华西的核型图进行了配置，可以直接使用。
  参数在karyotype类实例化时被读取，并以全局变量的形式使用。
  缺省使用，不需要在实例化时传入参数。

### 核型图图片文件的全路径

比如： "D:\\Prj\\github\\woodpecker\\test\\test_img\\L2311245727.001.K.TIF"
一个Karyotype实例对应一个核型图。

## 使用

```python
KYT_IMG_FP = "D:\\Prj\\github\\woodpecker\\test\\test_img\\L2311245727.001.K.TIF"
karyotype_chart = Karyotype(KYT_IMG_FP)
karyotype_chart.read_karyotype()

# 遍历获取每个染色体轮廓信息
for cntrs in karyotype_chart.chromo_cntr_dicts_orgby_cy.values():
    for idx, cntr in enumerate(cntrs):
        pass
"""
在执行完成read_karyotype()方法后，Karyotype 实例中 chromo_cntr_dicts_orgby_cy 属性的内容：
1. 每排的每个染色体轮廓都有染色体编号信息
2. 每排的每个染色体碎片轮廓都同其染色体主干轮廓合并了
3. 数据的排列格式是以cy为key的字典,每个cy对应的value是该cy行的所有的染色体轮廓信息，cy为该排染色体标号中心点的y坐标
   类似于: {
    248: [ {chromo_id:'1',chromo_idx:0,...}, {chromo_id:'2',chromo_idx:1,...},... ],
    468: [ {chromo_id:'6',chromo_idx:5,...}, {chromo_id:'7',chromo_idx:6,...},... ],
    629: [ {chromo_id:'11',chromo_idx:10,...}, {chromo_id:'12',chromo_idx:11,...},... ],
    812: [ {chromo_id:'19',chromo_idx:18,...}, {chromo_id:'20',chromo_idx:19,...},...]
   }
"""

# 另一种遍历方式获取每个染色体轮廓信息
for row_idx, cntrs in karyotype_chart.chromo_cntr_dicts.items():
    for cntr_idx, cntr in enumerate(cntrs):
        pass
"""
在执行完成read_karyotype()方法后，Karyotype 实例 chromo_cntr_dicts 属性的内容：
1. 每排的每个染色体轮廓都有染色体编号信息
2. 每排的每个染色体碎片轮廓都同其染色体主干轮廓合并了
3. 数据的排列格式是以list套list的形式，内层list为每排染色体轮廓的dict，外层list为所有排的list，排的顺序是从上到下的顺序
   类似于: [
    [ {chromo_id:'1',chromo_idx:0,...}, {chromo_id:'2',chromo_idx:1,...},... ],
    [ {chromo_id:'6',chromo_idx:5,...}, {chromo_id:'7',chromo_idx:6,...},... ],
    ...
    [ {chromo_id:'19',chromo_idx:18,...}, {chromo_id:'20',chromo_idx:19,...},...]
   ]
"""

```
